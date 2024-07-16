import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.preprocessing import normalize

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborSampler
from dataset import HierarchicalGraphDataset

def load_data(subset_size=None, verbose=True, hidden_dim=64):
    
    folder = "/extra/zhanglab0/xil43/HD/a/"
    file_path = f"{folder}binned_outputs/square_008um/filtered_feature_bc_matrix.h5"
    position_file_path = f"{folder}binned_outputs/square_008um/spatial/tissue_positions.parquet"
    super_node_file_path = f"{folder}square_008um_down.csv"
    processed_super_node_file_path = f"{folder}super_node_data.h5"
    reversed_file_path = f"{folder}square_008um_reversed.csv"
    
    # Load gene expression data
    adata = sc.read_10x_h5(file_path)
    adata.var_names_make_unique()
    gene_name = adata.var_names
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3")
    adata = adata[:, adata.var.highly_variable]
    # Data normalization and log transformation
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if verbose:
        print("AnnData object details:", adata)
        print(adata.var_names)
        print(adata.obs_names)
        
    # Only run once
    # super_node_data = pd.read_csv(super_node_file_path)
    # super_node_data = super_node_data.set_index('Unnamed: 0')
    # super_node_data.columns = gene_name
    # super_node_data.index.name = 'super node'
    # super_node_data.to_hdf(processed_super_node_file_path, key='super_node_data', mode='w')

    super_node_data = pd.read_hdf(processed_super_node_file_path, key='super_node_data')
    super_node_data = super_node_data.loc[:, adata.var_names]
    target_sum = 1e4
    super_node_data_normalized = normalize(super_node_data.values, norm='l1', axis=1) * target_sum
    super_node_data_normalized = pd.DataFrame(super_node_data_normalized, index=super_node_data.index, columns=super_node_data.columns)
    super_node_data_normalized = np.log1p(super_node_data_normalized)
    super_node_data = super_node_data_normalized
    if verbose:
        print("Super node gene expression data (top 3000 HVGs):\n", super_node_data.head())
        
    # Load position data
    position_data = pd.read_parquet(position_file_path)
    position_data.set_index('barcode', inplace=True)
    if verbose:
        print("Position data sample:\n", position_data.head())
        
    # Load super node to ordinary node mapping data
    super_to_normal = pd.read_csv(reversed_file_path, index_col='super node', header=None, names=['super node','node'])
    if verbose:
        print("Super node to ordinary node mapping sample:\n", super_to_normal.head())
        
    data = HierarchicalGraphDataset(adata, super_node_data, position_data, super_to_normal, subset_size=subset_size, verbose=verbose, hidden_dim=hidden_dim)
        
    return data

def load_parameter():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subset_size = 256
    verbose = True
    sizes = [-1]*2
    batch_size = 32
    input_dim = 3000
    hidden_dim = 64
    gnn_type = 'gcn'
    return device, subset_size, verbose, sizes, batch_size, input_dim, hidden_dim, gnn_type

def evaluate(model, data, device, batch_size=32):
    model.eval()
    criterion = torch.nn.MSELoss()
    total_mse_loss = 0
    total_nodes = 0

    sampler = NeighborSampler(edge_index=data.super_node_edge, sizes=[-1]*2, batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch_size, n_id, adjs in sampler:
            for i, super_node_id in enumerate(n_id):
                super_node = data.inverse_super_node_mapping[super_node_id.item()]
                subgraph = data.subgraphs[super_node]
                normal_node_hidden = torch.stack([data.normal_node_hidden[node] for node in subgraph.nodes]).to(device)

                # Get edge indices for normal nodes
                normal_edge_index = data.get_normal_node_edge_index(super_node).to(device)

                # Decode normal node hidden representations to gene expression
                node_gene_expression = F.relu(model.decoder_layer2(model.decoder_layer1(normal_node_hidden, normal_edge_index), normal_edge_index))

                # Get the real node gene expressions from the dataset
                real_node_gene_expression = torch.stack([
                    torch.tensor(subgraph.nodes[node]['features'].todense(), dtype=torch.float).to(device) 
                    if hasattr(subgraph.nodes[node]['features'], 'todense') 
                    else torch.tensor(subgraph.nodes[node]['features'], dtype=torch.float).to(device)
                    for node in subgraph.nodes
                ])

                # Ensure the real_node_gene_expression has the same shape as node_gene_expression
                real_node_gene_expression = real_node_gene_expression.squeeze()

                # Compute the MSE loss for this subgraph
                mse_loss = criterion(node_gene_expression, real_node_gene_expression)
                total_mse_loss += mse_loss.item() * len(subgraph.nodes)
                total_nodes += len(subgraph.nodes)

    # Compute the average MSE loss
    average_mse_loss = total_mse_loss / total_nodes
    return average_mse_loss






