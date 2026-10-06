import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from mp_api.client import MPRester
from jarvis.core.atoms import Atoms
from torch_geometric.data import Data, Batch

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.alignn import ALIGNN
from src.data.line_graph import ALIGNNTransform

def pymatgen_to_graph(structure, transform, cutoff_radius=5.0, max_neighbors=12):
    """Converts a Pymatgen Structure directly into an ALIGNN PyG Data object."""
    # Convert Pymatgen to Jarvis Atoms
    elements = [str(site.specie) for site in structure]
    coords = structure.cart_coords
    
    atoms = Atoms(
        lattice_mat=structure.lattice.matrix,
        coords=structure.frac_coords,
        elements=elements,
        cartesian=False
    )
    
    atomic_numbers = np.array(atoms.atomic_numbers)
    num_nodes = len(atomic_numbers)

    # Node Features
    x = torch.zeros((num_nodes, 100), dtype=torch.float)
    for i, z in enumerate(atomic_numbers):
        if 1 <= z <= 100:
            x[i, z - 1] = 1.0

    pos = torch.tensor(structure.cart_coords, dtype=torch.float)

    # Edge Features
    edge_src, edge_dst, edge_dist = [], [], []
    for i in range(num_nodes):
        dists = []
        for j in range(num_nodes):
            if i == j: continue
            diff = coords[i] - coords[j]
            d = np.linalg.norm(diff)
            if d <= cutoff_radius: dists.append((j, d))
            
        dists.sort(key=lambda item: item[1])
        for j, d in dists[:max_neighbors]:
            edge_src.append(i)
            edge_dst.append(j)
            edge_dist.append(d)

    if len(edge_src) == 0:
        return None

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_dist, dtype=torch.float).view(-1, 1)

    graph = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    return transform(graph)

def screen_database():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        print("Error: MP_API_KEY environment variable not set.")
        return

    print("1. Querying Materials Project for stable materials (e_above_hull < 0.05)...")
    with MPRester(api_key) as mpr:
        # Fetch summary docs for stable materials
        docs = mpr.materials.summary.search(
            energy_above_hull=(0, 0.05),
            fields=["material_id", "formula_pretty", "structure"]
        )
    
    print(f"Found {len(docs)} highly stable structures.")
    
    print("2. Loading ALIGNN Model...")
    model = ALIGNN(node_in_dim=100, edge_in_dim=1, angle_in_dim=1, hidden_dim=128, num_layers=4, out_dim=2)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    ckpt_path = os.path.join(project_root, "checkpoints", "best_alignn.pt")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Strip DDP prefix
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    transform = ALIGNNTransform()

    results = []
    batch_size = 64
    current_batch_graphs = []
    current_batch_meta = []

    print("3. Running High-Throughput Inference...")
    for i, doc in enumerate(tqdm(docs)):
        try:
            graph = pymatgen_to_graph(doc.structure, transform)
            if graph is None:
                continue
            current_batch_graphs.append(graph)
            current_batch_meta.append((doc.material_id, doc.formula_pretty))
        except Exception:
            continue
        
        current_batch_graphs.append(graph)
        current_batch_meta.append((doc.material_id, doc.formula_pretty))
        
        # Process in batches to fully utilize the L4 GPU
        if len(current_batch_graphs) == batch_size or i == len(docs) - 1:
            batch = Batch.from_data_list(current_batch_graphs).to(device)
            with torch.no_grad():
                preds = model(batch).cpu().numpy()
            
            for j, (mp_id, formula) in enumerate(current_batch_meta):
                k_pred, g_pred = preds[j]
                
                pugh = k_pred / g_pred if g_pred > 0 else 0
                hv = 0.92 * ((g_pred / k_pred) ** 1.137) * (g_pred ** 0.708) if (k_pred > 0 and g_pred > 0) else 0
                
                results.append({
                    "MP_ID": mp_id,
                    "Formula": formula,
                    "Bulk_Modulus_GPa": round(k_pred, 2),
                    "Shear_Modulus_GPa": round(g_pred, 2),
                    "Pugh_Ratio": round(pugh, 2),
                    "Vickers_Hardness_GPa": round(hv, 2)
                })
                
            current_batch_graphs = []
            current_batch_meta = []

    # Save to CSV
    df = pd.DataFrame(results)
    df = df.sort_values(by="Vickers_Hardness_GPa", ascending=False)
    df.to_csv("mp_alignn_screening_results.csv", index=False)
    
    print("\n" + "="*50)
    print(" SCREENING COMPLETE")
    print(f" Saved {len(df)} predictions to mp_alignn_screening_results.csv")
    print(" Top 5 Superhard Candidates:")
    print(df.head(5).to_string(index=False))
    print("="*50 + "\n")

if __name__ == "__main__":
    screen_database()
