import sys
import numpy as np
from motif import *
class sub_graph(object):
    def __init__(self, graph, hist_len, neg_size, directed=False, transform=None):
        self.hist_len = hist_len
        self.neg_size = neg_size
        self.directed = directed
        self.transform = transform
        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e2)
        self.node2hist = dict()  # 全图的邻接矩阵
        self.node2hist_tar = dict()
        self.graph = graph
        self.node_set = self.graph.node
        self.node_dim = len(self.node_set)
        self.node_map = self.graph.node_dic
        self.degrees = self.graph.degrees
        self.max_time = -sys.maxsize
        self.max_d_time = sys.maxsize  # Time interval [0, T]
        if len(self.node_set) != 0:
            self.neg_table = self.init_neg_table(self.degrees, self.node_dim, list(self.node_set))

        graph_idx = {}
        graph_idx1 = {}
        graph_idx2 = {}  # 为起始节点的矩阵
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

            graph_idx1[index] = [len(graph_idx2[edge.start_node]) - 1, len(graph_idx2[edge.end_node]) - 1]

            if edge.end_node + '_' + edge.start_node in graph_idx:
                for j in graph_idx[edge.end_node + '_' + edge.start_node]:
                    if edge.time == self.graph.edge[j].time:
                        time_count[index] = 2
                        time_count[j] = 2

        self.motif_p = motif(self.graph.edge, self.node_map, self.node_map, graph_idx, graph_idx2, graph_idx1,
                             time_count)

    def init_neg_table(self, degrees, node_dim, node_map):
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

    def sample(self, idx):
        row = self.graph.edge[idx]
        t_time = row.time
        s_node = self.node_map[row.start_node]
        t_node = self.node_map[row.end_node]

        hist_nodes, hist_times, type_all, h_t_time, h_t_masks, h_l_masks = self.motif_p.type(row, idx, self.hist_len)

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
            'history_types': np_h_type,
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

    def get_item(self):
        all = {
            'source_node': [],
            'target_node': [],
            'target_time': [],
            'neg_nodes': [],
            'history_nodes': [],
            'history_times': [],
            'history_types': [],
            'history_target_times': [],
            'history_masks': [],
            'history_t_masks': [],
        }
        idx = [i for i in range(len(self.graph.edge))]
        while len(idx) >= 1:
            index = random.choice(idx)
            idx.remove(index)
            row = self.sample(index)
            for i in all:
                all[i].append(row[i])
        for i in all:
            all[i] = np.array(all[i])
        return all


class _edge(object):
    def __init__(self, start_node,end_node,time,att):
        self.start_node = start_node
        self.end_node = end_node
        self.time = time
        self.att = att

class _graph(object):
    def __init__(self):
        self.node = set()
        self.edge = list()
        self.node_dic = dict()
        self.edge_count = 0
        self.degrees = dict()
        self.edge_att = dict()

    def add(self,edge):
        self.edge.append(edge)
        if edge.start_node not in self.node:
            self.node_dic[edge.start_node] = len(self.node)
            self.node.add(edge.start_node)
            self.degrees[edge.start_node] = 1
        else:
            self.degrees[edge.start_node] += 1
        if edge.end_node not in self.node:
            self.node_dic[edge.end_node] = len(self.node)
            self.node.add(edge.end_node)
            self.degrees[edge.end_node] = 1
        else:
            self.degrees[edge.end_node] += 1
        self.edge_count += 1
        self.edge_att[len(self.edge)-1] = edge.att

    def dis_g(self):
        return len(self.node), self.edge_count

class _datatset(object):
    def __init__(self,G,snap_len, hist_len, neg_size):
        self.graph = _graph()
        self.nx_graph = G
        self.max_time = -sys.maxsize
        self.max_d_time = sys.maxsize  # Time interval [0, T]
        self.snap_len = snap_len
        self.hist_len = hist_len
        self.neg_size = neg_size
        self.parse()

    def parse(self):
        for edge in self.nx_graph.edges(data=True):
            timestamp = edge[-1]['timestamp']
            self.max_time = max(timestamp, self.max_time)
            self.max_d_time = min(timestamp, self.max_d_time)
        self.slice_data()

    def slice_data(self):
        subgraph_len = int((self.max_time - self.max_d_time) / (self.snap_len * 60))
        self.all_graph = [_graph() for j in range(subgraph_len+1)]
        for edge in self.nx_graph.edges(data=True):
            timestamp = edge[-1]['timestamp']
            del edge[-1]['timestamp']
            time = (timestamp - self.max_d_time) / (self.max_time - self.max_d_time)
            self.all_graph[int((timestamp - self.max_d_time) / (self.snap_len * 60))].add(_edge(edge[0], edge[1], time, edge[-1]))
        i = 0
        while i < len(self.all_graph):
            nod_nums, edges_nums = self.all_graph[i].dis_g()
            if edges_nums == 0:
                del self.all_graph[i]
            else:
                self.all_graph[i] = sub_graph(self.all_graph[i],self.hist_len,self.neg_size)
                i += 1
        print(self.all_graph)
        self.snapshot_nums = len(self.all_graph)