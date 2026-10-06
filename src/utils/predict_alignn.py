import os
import sys
import torch
import numpy as np
import argparse
from jarvis.core.atoms import Atoms
from torch_geometric.data import Data, Batch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.alignn import ALIGNN
from src.data.line_graph import ALIGNNTransform

def strip_ddp_prefix(state_dict):
    """Removes the 'module.' prefix added by DistributedDataParallel."""
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def poscar_to_graph(poscar_path, cutoff_radius=5.0, max_neighbors=12):
    """Converts a POSCAR to an ALIGNN dual-graph."""
    atoms = Atoms.from_poscar(poscar_path)
    coords = np.array(atoms.cart_coords)
    atomic_numbers = np.array(atoms.atomic_numbers)
    num_nodes = len(atomic_numbers)

    # 1. Node Features (100-dim one-hot)
    x = torch.zeros((num_nodes, 100), dtype=torch.float)
    for i, z in enumerate(atomic_numbers):
        if 1 <= z <= 100:
            x[i, z - 1] = 1.0

    pos = torch.tensor(coords, dtype=torch.float)

    # 2. Edge Features (Distances)
    edge_src, edge_dst, edge_dist = [], [], []
    for i in range(num_nodes):
        dists = []
        for j in range(num_nodes):
            if i == j:
                continue
            diff = coords[i] - coords[j]
            d = np.linalg.norm(diff)
            if d <= cutoff_radius:
                dists.append((j, d))
        
        dists.sort(key=lambda item: item[1])
        for j, d in dists[:max_neighbors]:
            edge_src.append(i)
            edge_dst.append(j)
            edge_dist.append(d)

    if len(edge_src) == 0:
        raise ValueError("No edges found. Increase cutoff radius.")

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_dist, dtype=torch.float).view(-1, 1)

    graph = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

    # 3. Apply ALIGNN Transform (Generates angles and line-graph)
    transform = ALIGNNTransform()
    graph = transform(graph)
    
    return graph

def predict(poscar_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading POSCAR: {poscar_path}")
    graph = poscar_to_graph(poscar_path)
    batch = Batch.from_data_list([graph]).to(device)

    print(f"Loading Model: {checkpoint_path}")
    model = ALIGNN(node_in_dim=100, edge_in_dim=1, angle_in_dim=1, hidden_dim=128, num_layers=4, out_dim=2)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = strip_ddp_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    with torch.no_grad():
        preds = model(batch)
        k_pred, g_pred = preds[0].cpu().numpy()

    # Calculate Derived Properties
    pugh = k_pred / g_pred if g_pred > 0 else 0
    hv = 0.0
    if k_pred > 0 and g_pred > 0:
        hv = 0.92 * ((g_pred / k_pred) ** 1.137) * (g_pred ** 0.708)

    print("\n" + "="*50)
    print(" ALIGNN INFERENCE RESULTS")
    print("="*50)
    print(f"Bulk Modulus (K)   : {k_pred:.2f} GPa")
    print(f"Shear Modulus (G)  : {g_pred:.2f} GPa")
    print("-" * 50)
    print(f"Pugh's Ratio (K/G) : {pugh:.2f} (>1.75 = Ductile)")
    print(f"Vickers Hardness   : {hv:.2f} GPa")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict mechanical properties from a POSCAR using ALIGNN")
    parser.add_argument("poscar", type=str, help="Path to the VASP POSCAR file")
    parser.add_argument("--ckpt", type=str, default="MECNET_github/checkpoints/best_alignn.pt", help="Path to the trained model checkpoint")
    args = parser.parse_args()
    
    predict(args.poscar, args.ckpt)
