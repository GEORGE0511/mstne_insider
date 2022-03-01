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
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.directed = directed
        self.transform = transform
        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e8)
        self.node2hist = dict()     #全图的邻接矩阵
        self.node2hist_tar = dict()
        self.datasets = _datatset(nx.read_gpickle("../our_data/graph.gpickle2"))
        self.graph = self.datasets.graph
        self.node_set = self.graph.node
        self.node_dim = len(self.node_set)
        self.neg_node_map = self.graph.node_dic
        self.node_map = self.graph.node_dic
        self.degrees = self.graph.degrees
        self.max_time = -sys.maxsize
        self.max_d_time = sys.maxsize  # Time interval [0, T]
        for i in self.graph.edge:
            self.max_time = max(i.time,self.max_time)
            self.max_d_time = min(i.time,self.max_d_time)
        for i in self.graph.edge:
            i.time = (i.time - self.max_d_time) / (self.max_time - self.max_d_time)

        if len(self.node_set) != 0:
            self.neg_table = self.init_neg_table(self.degrees,self.node_dim,list(self.node_set))

        graph_idx = {}
        graph_idx1 = {}
        graph_idx2 = {}
        time_count = {}
        index = -1
        for edge in self.graph.edge:
            index += 1
            if edge.start_node + '_' + edge.end_node not in graph_idx:
                graph_idx[edge.start_node + '_' + edge.end_node] = []
            graph_idx[edge.start_node + '_' + edge.end_node].append(index)

            if edge.start_node not in graph_idx2:
                graph_idx2[edge.start_node] = []
            graph_idx2[edge.start_node].append(index)

            if edge.end_node not in graph_idx2:
                graph_idx2[edge.end_node] = []
            graph_idx2[edge.end_node].append(index)

            graph_idx1[index] = [len(graph_idx2[edge.start_node])-1,len(graph_idx2[edge.end_node])-1]

            if edge.end_node + '_' + edge.start_node in graph_idx:
                for j in graph_idx[edge.end_node + '_' + edge.start_node]:
                    if edge.time == self.graph.edge[j].time:
                        time_count[index] = 2
                        time_count[j] = 2

        self.motif_p = motif(self.graph.edge, self.node_map, self.neg_node_map, graph_idx, graph_idx2, graph_idx1, time_count)


    def init_neg_table(self,degrees,node_dim,node_map):
        neg_table = np.zeros((self.neg_table_size,))
        tot_sum, cur_sum, por = 0., 0., 0.
        n_id = 0
        for k in range(node_dim):
            tot_sum += np.power(degrees[node_map[k]], self.NEG_SAMPLING_POWER)
        for k in range(self.neg_table_size):
            if (k + 1.) / self.neg_table_size > por:
                cur_sum += np.power(degrees[node_map[n_id]], self.NEG_SAMPLING_POWER)
                por = cur_sum / tot_sum
                n_id += 1
            neg_table[k] = n_id - 1
        return neg_table

    def get_node_dim(self):
        return self.node_dim

    def __len__(self):
        return len(self.graph.edge)
        # return 1

    def __getitem__(self, idx):
        row = self.graph.edge[idx]
        t_time = row.time
        s_node = self.node_map[row.start_node]
        t_node = self.node_map[row.end_node]

        hist_nodes,hist_times,type_all,h_t_time,h_t_masks,h_l_masks = self.motif_p.type(row,idx,self.hist_len)

        np_h_nodes = np.zeros((self.hist_len,))
        np_h_nodes[:len(hist_nodes)] = hist_nodes
        np_h_times = np.zeros((self.hist_len,))
        np_h_times[:len(hist_times)] = hist_times
        np_h_type = np.zeros((self.hist_len,))
        np_h_type[:len(type_all)] = type_all
        np_h_t_time = np.zeros((self.hist_len,))
        np_h_t_time[:len(h_t_time)] = h_t_time
        np_h_masks = np.zeros((self.hist_len,))
        np_h_masks[:len(hist_nodes)] = h_l_masks[:len(hist_nodes)]
        np_h_t_masks = np.zeros((self.hist_len,))
        np_h_t_masks[:len(hist_nodes)] = h_t_masks[:len(hist_nodes)]

        # neg = list(self.node_set - hist_node)
        np_neg_nodes = self.negative_sampling()
        neg_node2 = []
        for i in range(len(np_neg_nodes)):
            neg_node2.append(np_neg_nodes[i])
        neg_node2 = np.array(neg_node2)

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'neg_nodes': neg_node2,
            'history_nodes': np_h_nodes,
            'history_times': np_h_times,
            'history_types':np_h_type,
            'history_target_times': np_h_t_time,
            'history_masks': np_h_masks,
            'history_t_masks': np_h_t_masks,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def negative_sampling(self):
        rand_idx = np.random.randint(0, self.neg_table_size, self.neg_size)
        sampled_nodes = []
        for j in self.neg_table[rand_idx]:
            sampled_nodes.append(int(j))
        return np.array(sampled_nodes)
