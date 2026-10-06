import os
import torch
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import sys

# Ensure MECNet can be imported
sys.path.append("/blue/hennig/ssavarkar/Ultra-hard-materials/MECNET_github")
from src.models.mecnet import MECNet

class JarvisDataset(InMemoryDataset):
    def __init__(self, save_path):
        super().__init__(root=os.path.dirname(save_path))
        self.data, self.slices = torch.load(save_path, weights_only=False)

def classification_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Explicit absolute paths
    ckpt_path = "/blue/hennig/ssavarkar/Ultra-hard-materials/MECNET_github/checkpoints/best_mecnet.pt"
    data_path = "/blue/hennig/ssavarkar/Ultra-hard-materials/MECNET_github/data/processed/jarvis_elastic_graphs.pt"
    
    print("Loading checkpoint and dataset...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MECNet(node_in_dim=100, edge_in_dim=1, scalar_in_dim=3, hidden_dim=128, out_dim=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dataset = JarvisDataset(save_path=data_path)
    test_loader = DataLoader(dataset[-3000:], batch_size=128, shuffle=False)
    
    true_ductile, pred_ductile = [], []
    true_hard, pred_hard = [], []
    
    # List to store our target materials
    candidates = []
    
    print("Evaluating materials...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch) # Raw GPa
            
            for i in range(len(batch.y)):
                k_pred, g_pred = preds[i, 0].item(), preds[i, 1].item()
                k_true, g_true = batch.y[i, 0].item(), batch.y[i, 1].item()
                
                if g_true <= 0 or g_pred <= 0 or k_true <= 0 or k_pred <= 0:
                    continue
                
                # Metrics
                pugh_pred = k_pred / g_pred
                pugh_true = k_true / g_true
                hv_pred = 0.92 * ((g_pred / k_pred) ** 1.137) * (g_pred ** 0.708)
                hv_true = 0.92 * ((g_true / k_true) ** 1.137) * (g_true ** 0.708)
                
                # Ductility Classification (Pugh > 1.75 is Ductile)
                true_ductile.append(1 if pugh_true > 1.75 else 0)
                pred_ductile.append(1 if pugh_pred > 1.75 else 0)
                
                # Hardness Classification (Hv > 20 GPa is "Hard")
                true_hard.append(1 if hv_true > 20.0 else 0)
                pred_hard.append(1 if hv_pred > 20.0 else 0)

                # Catch the overlap! Ductile AND Superhard
                if pugh_pred > 1.75 and hv_pred >= 20.0:
                    candidates.append({
                        "jid": batch.jid[i],
                        "predicted_K": round(k_pred, 2),
                        "predicted_G": round(g_pred, 2),
                        "predicted_Pugh": round(pugh_pred, 2),
                        "predicted_Hv": round(hv_pred, 2),
                        "true_Hv": round(hv_true, 2)
                    })

    # Export the JSON
    json_path = "/blue/hennig/ssavarkar/Ultra-hard-materials/MECNET_github/data/processed/superhard_ductile_candidates.json"
    with open(json_path, "w") as f:
        json.dump(candidates, f, indent=4)
    print(f"--> Extracted {len(candidates)} Superhard & Ductile candidate materials to: {json_path}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_ductile = confusion_matrix(true_ductile, pred_ductile)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_ductile, display_labels=['Brittle', 'Ductile'])
    disp1.plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title("Ductility Screening (Pugh Threshold)")
    
    cm_hard = confusion_matrix(true_hard, pred_hard)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_hard, display_labels=['Soft', 'Hard (≥20 GPa)'])
    disp2.plot(ax=axes[1], cmap='Reds', colorbar=False)
    axes[1].set_title("Superhard Screening (Vickers Threshold)")
    
    plt.tight_layout()
    plot_path = "/blue/hennig/ssavarkar/Ultra-hard-materials/MECNET_github/screening_confusion_matrices.png"
    plt.savefig(plot_path, dpi=300)
    print(f"--> Saved classification evaluation plot to: {plot_path}")

if __name__ == "__main__":
    classification_evaluation()
