import os
import sys
import json
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

# Add root to path to import MECNet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.mecnet import MECNet

class JarvisDataset(InMemoryDataset):
    def __init__(self, save_path):
        super().__init__(root=os.path.dirname(save_path))
        self.data, self.slices = torch.load(save_path, weights_only=False)

def find_hard_and_ductile_materials():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints/best_mecnet.pt"))
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed/jarvis_elastic_graphs.pt"))
    output_json = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed/hard_ductile_candidates.json"))
    
    # Load model
    print("Loading MECNet checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MECNet(node_in_dim=100, edge_in_dim=1, scalar_in_dim=3, hidden_dim=128, out_dim=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    print("Loading dataset for screening...")
    dataset = JarvisDataset(save_path=data_path)
    
    # We will scan the whole dataset to find all possible candidates for your future DFT work
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    candidates = []
    
    print("Scanning for materials with Hv >= 20 GPa AND K/G > 1.75...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)
            
            for i in range(len(batch.y)):
                k_pred = preds[i, 0].item()
                g_pred = preds[i, 1].item()
                k_true = batch.y[i, 0].item()
                g_true = batch.y[i, 1].item()
                jid = batch.jid[i]
                
                # Skip unphysical predictions
                if k_pred <= 0 or g_pred <= 0:
                    continue
                    
                pugh_pred = k_pred / g_pred
                hv_pred = 0.92 * ((g_pred / k_pred) ** 1.137) * (g_pred ** 0.708)
                
                # The Golden Criteria: Superhard AND Ductile
                if hv_pred >= 20.0 and pugh_pred > 1.75:
                    
                    # Calculate ground truths for reference
                    if k_true > 0 and g_true > 0:
                        pugh_true = k_true / g_true
                        hv_true = 0.92 * ((g_true / k_true) ** 1.137) * (g_true ** 0.708)
                    else:
                        pugh_true, hv_true = 0.0, 0.0

                    candidates.append({
                        "jid": jid,
                        "predicted_physics": {
                            "Hv_GPa": round(hv_pred, 2),
                            "Pugh_ratio": round(pugh_pred, 2),
                            "K_GPa": round(k_pred, 2),
                            "G_GPa": round(g_pred, 2)
                        },
                        "true_physics": {
                            "Hv_GPa": round(hv_true, 2),
                            "Pugh_ratio": round(pugh_true, 2),
                            "K_GPa": round(k_true, 2),
                            "G_GPa": round(g_true, 2)
                        }
                    })

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(candidates, f, indent=4)
        
    print(f"\nDiscovery complete! Found {len(candidates)} candidate materials.")
    print(f"Candidate structures and properties saved to: {output_json}")

if __name__ == "__main__":
    find_hard_and_ductile_materials()
