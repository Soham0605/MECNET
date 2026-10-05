import os
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from mecnet import MECNet

class JarvisDataset(InMemoryDataset):
    def __init__(self, save_path):
        super().__init__(root=os.path.dirname(save_path))
        self.data, self.slices = torch.load(save_path, weights_only=False)

if __name__ == "__main__":
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed/jarvis_elastic_graphs.pt"))
    dataset = JarvisDataset(save_path=data_path)
    loader = DataLoader(dataset[:4], batch_size=4, shuffle=False)
    
    batch = next(iter(loader))
    model = MECNet()
    output = model(batch)
    
    print(f"Batch loaded: {batch.num_graphs} crystal graphs")
    print(f"Model output shape: {output.shape} (Expected: [4, 2])")
    print(f"Sample prediction [K, G]:\n{output.detach().numpy()}")
    assert output.shape == (4, 2), "Output dimension mismatch!"
    print("MECNet forward pass verified successfully.")
