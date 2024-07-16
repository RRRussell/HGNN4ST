import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_type='gcn'):
        super(GraphAutoencoder, self).__init__()
        # Select GNN type
        if gnn_type == 'gcn':
            self.gnn_layer = GCNConv
        elif gnn_type == 'gat':
            self.gnn_layer = GATConv
        elif gnn_type == 'graphsage':
            self.gnn_layer = SAGEConv
        elif gnn_type == 'graphtransformer':
            self.gnn_layer = TransformerConv
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

        self.encoder_layer1 = self.gnn_layer(input_dim, hidden_dim)
        self.encoder_layer2 = self.gnn_layer(hidden_dim, hidden_dim)

        self.decoder_layer1 = self.gnn_layer(hidden_dim, hidden_dim)
        self.decoder_layer2 = self.gnn_layer(hidden_dim, input_dim)

        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x, edge_indices):
        edge_index_1 = edge_indices[0]
        edge_index_2 = edge_indices[1]
        x = self.dropout(x)
        x = F.elu(self.encoder_layer1(x, edge_index_1))
        z = F.elu(self.encoder_layer2(x, edge_index_2))  # hidden representation
        x = F.elu(self.decoder_layer1(z, edge_index_1))
        x = F.relu(self.decoder_layer2(x, edge_index_2))
        return z, x

class NoisePredictor(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(NoisePredictor, self).__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def custom_readout(node_hidden):
    return torch.mean(node_hidden, dim=0)
