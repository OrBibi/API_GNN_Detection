# src/model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNClassifier(torch.nn.Module):
    """
    Graph Neural Network (GNN) model for classification of API logs (Graph-Level Classification).
    """
    def __init__(self, feature_dim: int = 64, hidden_channels: int = 64, num_classes: int = 2):
        super().__init__()
        
        # 1. GCN Layers (Feature Aggregation)
        # GCNConv aggregates features from neighbors
        self.conv1 = GCNConv(feature_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # 2. Linear Layers (Classification Head)
        # The hidden_channels size after pooling is used for the input of the linear layers
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = torch.nn.Linear(hidden_channels // 2, num_classes)

    def forward(self, x, edge_index, batch_index):
        # x: Node feature matrix [num_nodes, feature_dim]
        # edge_index: Graph connectivity [2, num_edges]
        # batch_index: Batch assignment vector [num_nodes]
        
        # 1. Message Passing Layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 2. Readout/Pooling Layer
        # global_mean_pool aggregates node features to a single graph representation
        # x shape changes from [num_nodes, hidden_channels] to [batch_size, hidden_channels]
        x = global_mean_pool(x, batch_index)
        
        # 3. Classification Head
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    # Simple model test
    model = GNNClassifier()
    print("GNN Model defined successfully.")