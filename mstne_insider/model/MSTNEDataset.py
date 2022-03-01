from torch.utils.data import Dataset
import numpy as np
import sys
from motif import *
import networkx as nx
from datatable import dt
from dataset import _datatset
from dataset import _graph
from dataset import _edge

class MSTNEDataSet(Dataset):
    def __init__(self, neg_size, hist_len, directed=False, transform=None):
        self.datasets = _datatset(nx.read_gpickle("../our_data/graph.gpickle2"), 60, hist_len, neg_size)
        self.all_graph = self.datasets.all_graph

    def __len__(self):
        return len(self.all_graph)

    def __getitem__(self, idx):
        subgraph = self.all_graph[idx]
        item = subgraph.get_item()
        item['idx'] = idx
        return item

    def get_node_dim(self):
        return len(self.datasets.nx_graph.nodes())





