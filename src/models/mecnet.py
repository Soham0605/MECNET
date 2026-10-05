import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, global_mean_pool

class MECNet(nn.Module):
    def __init__(self, node_in_dim=100, edge_in_dim=1, scalar_in_dim=3, hidden_dim=128, out_dim=2):
        super(MECNet, self).__init__()
        
        # 1. Edge feature expansion networks for NNConv
        edge_net1 = nn.Sequential(
            nn.Linear(edge_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, node_in_dim * hidden_dim)
        )
        edge_net2 = nn.Sequential(
            nn.Linear(edge_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim * hidden_dim)
        )
        edge_net3 = nn.Sequential(
            nn.Linear(edge_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim * hidden_dim)
        )
        
        # 2. Graph Message Passing Layers
        self.conv1 = NNConv(node_in_dim, hidden_dim, edge_net1, aggr='mean')
        self.conv2 = NNConv(hidden_dim, hidden_dim, edge_net2, aggr='mean')
        self.conv3 = NNConv(hidden_dim, hidden_dim, edge_net3, aggr='mean')
        
        # 3. Global Scalar Encoder: processes [Ef, Eg, form_energy]
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_in_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
            nn.SiLU()
        )
        
        # 4. Joint Multi-Modal Projection Head
        # Concatenates pooled graph representation (hidden_dim) + scalar representation (64)
        joint_dim = hidden_dim + 64
        self.mlp = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, out_dim)  # Output: [Bulk Modulus K, Shear Modulus G]
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Graph convolution branch
        x = torch.relu(self.conv1(x, edge_index, edge_attr))
        x = torch.relu(self.conv2(x, edge_index, edge_attr))
        x = torch.relu(self.conv3(x, edge_index, edge_attr))
        h_graph = global_mean_pool(x, batch)
        
        # Global scalar branch
        u = data.u
        if u.dim() == 1:
            u = u.unsqueeze(0)
        h_scalar = self.scalar_encoder(u)
        
        # Multimodal fusion and regression
        h_combined = torch.cat([h_graph, h_scalar], dim=1)
        out = self.mlp(h_combined)
        return out
