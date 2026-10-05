import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.mecnet import MECNet


class JarvisDataset(InMemoryDataset):
    def __init__(self, save_path):
        super().__init__(root=os.path.dirname(save_path))
        self.data, self.slices = torch.load(save_path, weights_only=False)


def setup_ddp():
    """Initializes NCCL backend across GPUs allocated by torchrun / SLURM."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def compute_target_stats(dataset, train_indices, device):
    """Computes mean and std of [K, G] for the training subset to normalize loss."""
    targets = []
    for idx in train_indices:
        targets.append(dataset[idx].y)
    targets = torch.cat(targets, dim=0).to(device)
    mean = targets.mean(dim=0, keepdim=True)
    std = targets.std(dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-5)
    return mean, std


def train():
    local_rank, global_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # Maximum throughput configurations for NVIDIA Ampere/Ada GPUs
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1. Dataset Loading & Partitioning
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/processed/jarvis_elastic_graphs.pt")
    )
    if global_rank == 0:
        print(f"[Rank 0] Loading dataset from: {data_path}")

    dataset = JarvisDataset(save_path=data_path)
    total_len = len(dataset)

    # Deterministic split: 80% train, 10% val, 10% test
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_len, generator=generator).tolist()
    train_split = int(0.80 * total_len)
    val_split = int(0.90 * total_len)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]

    train_subset = [dataset[i] for i in train_indices]
    val_subset = [dataset[i] for i in val_indices]

    # Precalculate normalization constants
    mean_y, std_y = compute_target_stats(dataset, train_indices, device)
    if global_rank == 0:
        print(f"[Rank 0] Target Means [K, G]: {mean_y.cpu().numpy()[0]}")
        print(f"[Rank 0] Target Stds  [K, G]: {std_y.cpu().numpy()[0]}")

    # 2. High-Throughput Distributed DataLoaders
    # Per-GPU batch size: 256 yields a combined effective batch size of 512 across 2 GPUs
    BATCH_SIZE_PER_GPU = 256
    NUM_WORKERS = 8

    train_sampler = DistributedSampler(
        train_subset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        drop_last=True
    )
    val_sampler = DistributedSampler(
        val_subset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False,
        drop_last=False
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    # 3. Model Setup & DDP Wrapping
    model = MECNet(node_in_dim=100, edge_in_dim=1, scalar_in_dim=3, hidden_dim=128, out_dim=2).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-5)
    criterion = nn.SmoothL1Loss()

    EPOCHS = 150
    best_val_loss = float("inf")
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints"))
    if global_rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"[Rank 0] Training initialized for {EPOCHS} epochs across {world_size} GPUs.\n")

    # 4. Training Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        total_train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # Native bfloat16 mixed precision on RTX 6000 Tensor Cores
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model(batch)
                norm_preds = (preds - mean_y) / std_y
                norm_targets = (batch.y - mean_y) / std_y
                loss = criterion(norm_preds, norm_targets)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()

        scheduler.step()

        # Validation Phase
        model.eval()
        val_k_mae, val_g_mae, total_val_loss = 0.0, 0.0, 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds = model(batch)
                    norm_preds = (preds - mean_y) / std_y
                    norm_targets = (batch.y - mean_y) / std_y
                    loss = criterion(norm_preds, norm_targets)

                total_val_loss += loss.item() * batch.num_graphs
                val_k_mae += torch.sum(torch.abs(preds[:, 0] - batch.y[:, 0])).item()
                val_g_mae += torch.sum(torch.abs(preds[:, 1] - batch.y[:, 1])).item()
                val_samples += batch.num_graphs

        # Aggregate metrics across both GPUs via NCCL All-Reduce
        metrics = torch.tensor(
            [total_train_loss, total_val_loss, val_k_mae, val_g_mae, float(val_samples)],
            device=device,
            dtype=torch.float32
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        global_train_loss = metrics[0].item() / (len(train_loader) * world_size)
        global_val_loss = metrics[1].item() / metrics[4].item()
        global_val_k_mae = metrics[2].item() / metrics[4].item()
        global_val_g_mae = metrics[3].item() / metrics[4].item()

        if global_rank == 0:
            print(
                f"Epoch {epoch:03d} | Train Loss: {global_train_loss:.4f} | "
                f"Val Loss: {global_val_loss:.4f} | "
                f"Val MAE_K: {global_val_k_mae:.2f} GPa | "
                f"Val MAE_G: {global_val_g_mae:.2f} GPa"
            )

            if global_val_loss < best_val_loss:
                best_val_loss = global_val_loss
                ckpt_path = os.path.join(checkpoint_dir, "best_mecnet.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mean_y": mean_y.cpu(),
                    "std_y": std_y.cpu(),
                    "val_loss": best_val_loss
                }, ckpt_path)
                print(f"  --> Saved new best checkpoint to {ckpt_path}")

    cleanup_ddp()


if __name__ == "__main__":
    train()
