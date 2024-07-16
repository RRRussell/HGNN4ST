import numpy as np
import networkx as nx
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torch.nn as nn

class HierarchicalGraphDataset(Dataset):
    def __init__(self, adata, super_node_data, position_data, super_to_normal, subset_size=None, verbose=False, hidden_dim=64):
        self.adata = adata
        self.super_node_data = super_node_data.iloc[:subset_size] if subset_size else super_node_data
        self.position_data = position_data
        self.super_to_normal = super_to_normal
        self.verbose = verbose
        self.hidden_dim = hidden_dim

        # Initialize mappings
        self.super_node_mapping = {}
        self.inverse_super_node_mapping = {}
        self.local_normal_node_mappings = {}

        self.main_graph, self.subgraphs, self.normal_node_edges = self.build_graphs()
        self.super_node_edge = self.get_super_node_edge_index()
        self.super_node_features = self.get_super_node_features()
        self.normal_node_hidden = self.initialize_normal_node_hidden()

    def build_graphs(self):
        main_graph = nx.Graph()
        subgraphs = {}
        normal_node_edges = {}

        super_node_iterator = tqdm(self.super_node_data.iterrows(), total=self.super_node_data.shape[0], desc="Processing super nodes") if self.verbose else self.super_node_data.iterrows()
        for index, row in super_node_iterator:
            parts = index.split('_')
            x, y = int(parts[0]), int(parts[1])

            main_graph.add_node(index, features=row.values, x=x, y=y)
            if index not in self.super_node_mapping:
                self.super_node_mapping[index] = len(self.super_node_mapping)
                self.inverse_super_node_mapping[self.super_node_mapping[index]] = index

            for offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = f'{x+offset[0]:05}_{y+offset[1]:05}'
                if neighbor in self.super_node_data.index:
                    main_graph.add_edge(index, neighbor)

            subgraph = nx.Graph()
            nodes = eval(self.super_to_normal.loc[index, 'node'])
            node_positions = {node: self.position_data.loc[node] for node in nodes if node in self.position_data.index}

            local_node_mapping = {}
            for local_id, (node, pos) in enumerate(node_positions.items()):
                local_node_mapping[node] = local_id
                subgraph.add_node(node, features=self.adata.X[self.adata.obs_names.get_loc(node), :], x=pos['pxl_row_in_fullres'], y=pos['pxl_col_in_fullres'])

            self.local_normal_node_mappings[index] = local_node_mapping

            subgraph_edges = []
            for i, node1 in enumerate(node_positions.keys()):
                for j in range(i + 1, len(node_positions)):
                    node2 = list(node_positions.keys())[j]
                    distance = np.linalg.norm([node_positions[node1]['pxl_row_in_fullres'] - node_positions[node2]['pxl_row_in_fullres'], node_positions[node1]['pxl_col_in_fullres'] - node_positions[node2]['pxl_col_in_fullres']])
                    if distance < 50:
                        subgraph.add_edge(node1, node2)
                        subgraph_edges.append((node1, node2))
                        subgraph.add_edge(node2, node1)
                        subgraph_edges.append((node2, node1))

            if self.verbose:
                if not nx.is_connected(subgraph):
                    print(f"Subgraph for super node {index} is not connected.")
                print(f"Subgraph for super node {index} has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")

            subgraphs[index] = subgraph
            normal_node_edges[index] = subgraph_edges

        return main_graph, subgraphs, normal_node_edges

    def get_super_node_edge_index(self):
        edge_index = torch.tensor([[self.super_node_mapping[u], self.super_node_mapping[v]] for u, v in self.main_graph.edges], dtype=torch.long).t().contiguous()
        return edge_index

    def get_super_node_features(self):
        features = [self.main_graph.nodes[node]['features'] for node in self.main_graph.nodes]
        features_tensor = torch.tensor(np.array(features), dtype=torch.float)
        return features_tensor

    def initialize_normal_node_hidden(self):
        normal_nodes = [node for subgraph in self.subgraphs.values() for node in subgraph.nodes]
        normal_node_hidden = nn.ParameterDict({node: nn.Parameter(torch.empty((self.hidden_dim,))) for node in normal_nodes})
        for param in normal_node_hidden.values():
            nn.init.xavier_uniform_(param.unsqueeze(0))  # Use unsqueeze to make it 2D
        return normal_node_hidden

    def get_normal_node_edge_index(self, super_node):
        if super_node not in self.normal_node_edges:
            raise ValueError(f"No normal node edges found for super node {super_node}")
        subgraph_edges = self.normal_node_edges[super_node]
        local_node_mapping = self.local_normal_node_mappings[super_node]
        edge_index = torch.tensor([[local_node_mapping[node1], local_node_mapping[node2]] for node1, node2 in subgraph_edges], dtype=torch.long).t().contiguous()
        return edge_index
