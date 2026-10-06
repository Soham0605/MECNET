import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.alignn import ALIGNN

class JarvisAlignnDataset(InMemoryDataset):
    def __init__(self, save_path):
        super().__init__(root=os.path.dirname(save_path))
        self.data, self.slices = torch.load(save_path, weights_only=False)

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def train():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    data_path = "/blue/hennig/ssavarkar/Ultra-hard-materials/MECNET_github/data/processed/jarvis_alignn_graphs.pt"
    if local_rank == 0:
        print(f"Loading ALIGNN dataset from: {data_path}")

    dataset = JarvisAlignnDataset(data_path)
    
    # 80/10/10 Split
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    train_data = dataset[:train_len]
    val_data = dataset[train_len:train_len + val_len]

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=local_rank, shuffle=False)

    # Batch size 32 per GPU to prevent CUDA OOM on line graphs
    train_loader = DataLoader(train_data, batch_size=16, sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=16, sampler=val_sampler, num_workers=2, pin_memory=True)

    # Initialize ALIGNN
    model = ALIGNN(
        node_in_dim=100,
        edge_in_dim=1,
        angle_in_dim=1,
        hidden_dim=128,
        num_layers=4,
        out_dim=2
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = torch.nn.L1Loss() # MAE Loss in GPa
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs = 150
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    save_dir = "/blue/hennig/ssavarkar/Ultra-hard-materials/MECNET_github/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_alignn.pt")

    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch)
            loss = criterion(preds, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch)
                loss = criterion(preds, batch.y)
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Reduce validation loss across all GPUs
        val_loss_tensor = torch.tensor(avg_val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        reduced_val_loss = val_loss_tensor.item()

        if local_rank == 0:
            print(f"Epoch {epoch:03d} | Train MAE: {avg_train_loss:.2f} GPa | Val MAE: {reduced_val_loss:.2f} GPa")
            if reduced_val_loss < best_val_loss:
                best_val_loss = reduced_val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "val_loss": best_val_loss
                }, best_model_path)
                print(f"  --> Saved new best ALIGNN model (Val MAE: {best_val_loss:.2f} GPa)")

    cleanup_ddp()

if __name__ == "__main__":
    train()
