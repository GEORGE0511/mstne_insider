import networkx as nx
import pickle
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
    def __init__(self,G,has_graph=False):
        self.graph = _graph()
        self.nx_graph = G
        if has_graph:
            self.graph = self.load_graph()
        else:
            self.parse()
            self.save_graph()

    def save_graph(self):
        output_hal = open("graph.pkl", 'wb')
        str = pickle.dumps(self.graph)
        output_hal.write(str)
        output_hal.close()

    def load_graph(self):
        with open("graph.pkl", 'rb') as file:
            return pickle.loads(file.read())

    def parse(self):
        for edge in self.nx_graph.edges(data=True):
            timestamp = edge[-1]['timestamp']
            del edge[-1]['timestamp']
            self.graph.add(_edge(edge[0], edge[1], timestamp, edge[-1]))
        print(self.graph.dis_g())
        # print(self.graph.node)