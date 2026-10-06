import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, LayerNorm, SiLU
from torch_geometric.nn import global_mean_pool, CGConv

class ALIGNNBlock(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 1. Line Graph Message Passing (Updates bonds using angles)
        self.edge_conv = CGConv(channels=hidden_dim, dim=hidden_dim, aggr='add')
        self.edge_norm = LayerNorm(hidden_dim)
        
        # 2. Standard Graph Message Passing (Updates atoms using newly updated bonds)
        self.node_conv = CGConv(channels=hidden_dim, dim=hidden_dim, aggr='add')
        self.node_norm = LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr, line_edge_index, angle_attr):
        # Step 1: Angle -> Bond (Skip Connection included)
        edge_attr_new = self.edge_conv(edge_attr, line_edge_index, angle_attr)
        edge_attr = edge_attr + F.silu(self.edge_norm(edge_attr_new))
        
        # Step 2: Bond -> Atom (Skip Connection included)
        x_new = self.node_conv(x, edge_index, edge_attr)
        x = x + F.silu(self.node_norm(x_new))
        
        return x, edge_attr

class ALIGNN(torch.nn.Module):
    def __init__(self, node_in_dim=100, edge_in_dim=1, angle_in_dim=1, hidden_dim=128, num_layers=4, out_dim=2):
        super().__init__()
        
        # Initial Embeddings (Projecting inputs to 128D latent space)
        self.node_embed = Sequential(Linear(node_in_dim, hidden_dim), SiLU())
        self.edge_embed = Sequential(Linear(edge_in_dim, hidden_dim), SiLU())
        self.angle_embed = Sequential(Linear(angle_in_dim, hidden_dim), SiLU())
        
        # ALIGNN Message Passing Layers
        self.layers = torch.nn.ModuleList([
            ALIGNNBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # Multi-task Readout (Predicts K and G)
        self.post_pool = Sequential(
            Linear(hidden_dim, hidden_dim),
            SiLU(),
            Linear(hidden_dim, out_dim)
        )

    def forward(self, data):
        # Extract Standard Graph
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Extract Line Graph
        line_edge_index, angle_attr = data.line_edge_index, data.angle_attr
        
        # Embed physical inputs
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        angle_attr = self.angle_embed(angle_attr)
        
        # Pass messages across alternating graphs
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr, line_edge_index, angle_attr)
            
        # Macroscopic Collapse (Average all atoms into one unit cell vector)
        out = global_mean_pool(x, data.batch)
        
        return self.post_pool(out)
