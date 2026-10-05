import os
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import sys

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.mecnet import MECNet

class JarvisDataset(InMemoryDataset):
    def __init__(self, save_path):
        super().__init__(root=os.path.dirname(save_path))
        self.data, self.slices = torch.load(save_path, weights_only=False)

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the trained checkpoint
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints/best_mecnet.pt"))
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 2. Initialize Model and apply weights
    model = MECNet(node_in_dim=100, edge_in_dim=1, scalar_in_dim=3, hidden_dim=128, out_dim=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    mean_y = checkpoint['mean_y'].to(device)
    std_y = checkpoint['std_y'].to(device)
    
    # 3. Load a sample of data (using the test split implicitly from the end of the dataset)
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed/jarvis_elastic_graphs.pt"))
    dataset = JarvisDataset(save_path=data_path)
    
    # Let's test the last 5 materials in the dataset
    test_loader = DataLoader(dataset[-5:], batch_size=1, shuffle=False)
    
    print("\n================ MECNET PREDICTIONS ================")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            
            # Forward pass
            norm_preds = model(batch)
            
            # Reverse the Z-score normalization
            preds = (norm_preds * std_y) + mean_y
            pred_k, pred_g = preds[0, 0].item(), preds[0, 1].item()
            
            # Ground truth
            true_k, true_g = batch.y[0, 0].item(), batch.y[0, 1].item()
            
            # Physics Inversion
            # 1. Pugh's Ratio (K/G)
            pugh_ratio = pred_k / pred_g if pred_g > 0 else 0
            behavior = "Ductile" if pugh_ratio > 1.75 else "Brittle"
            
            # 2. Vickers Hardness (Chen's Model)
            # Hv = 0.92 * (G/K)^1.137 * G^0.708
            if pred_g > 0 and pred_k > 0:
                hv = 0.92 * ((pred_g / pred_k) ** 1.137) * (pred_g ** 0.708)
            else:
                hv = 0.0
                
            print(f"Material {i+1} (JID: {batch.jid[0]})")
            print(f"  Predicted Moduli : K = {pred_k:.1f} GPa, G = {pred_g:.1f} GPa")
            print(f"  True Moduli      : K = {true_k:.1f} GPa, G = {true_g:.1f} GPa")
            print(f"  Derived Physics  : Pugh's Ratio = {pugh_ratio:.2f} ({behavior}) | Vickers Hardness = {hv:.1f} GPa")
            print("-" * 60)

if __name__ == "__main__":
    predict()
