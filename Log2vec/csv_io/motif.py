'''
Author: your name
Date: 2021-03-03 19:05:14
LastEditTime: 2021-03-18 16:36:22
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \mctne\motif.py
'''
import random

class motif(object):
    def __init__(self,all_edges_info, node_map, neg_node_map, graph_idx, graph_idx2, graph_idx1, time_count):
        self.all_edges_info = all_edges_info
        self.node_map = node_map
        self.neg_node_map = neg_node_map
        self.graph_idx_all = graph_idx
        self.graph_idx2_all = graph_idx2
        self.graph_idx1_all = graph_idx1
        self.time_count_all = time_count

    def find_type_MSTNE(self,left_time,right_time,l_type,r_type):
        if l_type == 1 and r_type == 1:
            tp = 3
        elif l_type == 0 and r_type == 0:
            tp = 4
        elif l_type == 0 and r_type == 1:
            tp = 5
        elif l_type == 1 and r_type == 0:
            tp = 6
        elif l_type == 1 and r_type == 2:
            tp = 7
        elif l_type == 0 and r_type == 2:
            tp = 8
        elif l_type == 2 and r_type == 0:
            tp = 9
        elif l_type == 2 and r_type == 1:
            tp = 10
        elif l_type == 2 and r_type == 2:
            tp = 11
        if left_time < right_time:
            return 2 * tp +1
        else:
            return 2 * tp

    def type(self,all_edges,index,hist_len):
        motif_edges = []
        t_nodes = []
        type_all = []
        hist_nodes = []
        hist_time = []
        h_t_time = []
        hist_time_idx = []
        h_t_time_idx = []

        h_t_masks = [0 for i in range(hist_len)]
        h_l_masks = [0 for i in range(hist_len)]

        next_edges = all_edges
        s_node = next_edges.start_node
        t_node = next_edges.end_node
        time = next_edges.time
        
        inx = 0
        while len(hist_nodes)  < hist_len:
            try:
                l_neighbors = self.graph_idx2_all[s_node][0:self.graph_idx1_all[index][0]]
                r_neighbors = self.graph_idx2_all[t_node][0:self.graph_idx1_all[index][1]]
            except Exception as e:
                print(e)

            l_counts = len(l_neighbors)
            r_counts = len(r_neighbors)

            if l_counts + r_counts == 0:
                break

            index_2 = random.sample(list(range(l_counts + r_counts)),1)[0]

            l_type = -1
            if index_2 <= l_counts - 1:
                a_index = l_neighbors[index_2]
                if a_index in self.time_count_all:
                    l_type = 2
                l_s_node = self.all_edges_info[a_index].start_node
                l_t_node = self.all_edges_info[a_index].end_node
                l_time = self.all_edges_info[a_index].time

            else:
                a_index = r_neighbors[index_2-l_counts]
                if a_index in self.time_count_all:
                    l_type = 2
                l_s_node = self.all_edges_info[a_index].start_node
                l_t_node = self.all_edges_info[a_index].end_node
                l_time = self.all_edges_info[a_index].time
            
            one_type = -1
            if l_s_node == s_node:
                one_type = 0
                if l_type == -1:
                    l_type = 0
                if (str(l_t_node)+ '_' + str(t_node)) in self.graph_idx_all:
                    cou = self.graph_idx_all[str(l_t_node)+ '_' + str(t_node)]
                    b = [i for i in cou if i <= index]
                    if len(b) != 0:
                        r_index = random.sample(b,1)[0]
                        r_time = self.all_edges_info[r_index].time
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 1
                        motif_edges.append([l_t_node,l_time,r_time,self.find_type_MSTNE(l_time,r_time,l_type,r_type)])

                elif (str(t_node)+ '_' + str(l_t_node)) in self.graph_idx_all:
                    cou2 = self.graph_idx_all[str(t_node)+ '_' + str(l_t_node)]
                    b = [i for i in cou2 if i <= index]
                    if len(b) != 0:
                        r_index = random.sample(b,1)[0]
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 0
                        r_time = self.all_edges_info[r_index].time
                        motif_edges.append([l_t_node,l_time,r_time,self.find_type_MSTNE(l_time,r_time,l_type,r_type)])

            elif l_t_node == s_node:
                one_type = 1
                if l_type == -1:
                    l_type = 1

                if (str(l_s_node)+ '_' + str(t_node)) in self.graph_idx_all:
                    cou = self.graph_idx_all[str(l_s_node)+ '_' + str(t_node)]
                    b = [i for i in cou if i <= index]
                    if len(b) != 0:
                        r_index = random.sample(b,1)[0]
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 1
                        r_time = self.all_edges_info[r_index].time
                        motif_edges.append([l_s_node,l_time,r_time,self.find_type_MSTNE(l_time,r_time,l_type,r_type)])

                if (str(t_node)+ '_' + str(l_s_node)) in self.graph_idx_all:
                    cou2 = self.graph_idx_all[str(t_node)+ '_' + str(l_s_node)]
                    b = [i for i in cou2 if i <= index]
                    if len(b) != 0:
                        r_index = random.sample(b,1)[0]
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 0

                        r_time = self.all_edges_info[r_index].time
                        motif_edges.append([l_s_node,l_time,r_time,self.find_type_MSTNE(l_time,r_time,l_type,r_type)])
            
            elif l_s_node == t_node:
                one_type = 2
                if l_type == -1:
                    l_type = 0
                if (str(l_t_node)+ '_' + str(s_node)) in self.graph_idx_all:
                    cou = self.graph_idx_all[str(l_t_node)+ '_' + str(s_node)]
                    b = [i for i in cou if i <= index]
                    if len(b) != 0:
                        r_index = random.sample(b,1)[0]
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 1
                        r_time = self.all_edges_info[r_index].time
                        motif_edges.append([l_t_node,r_time,l_time,self.find_type_MSTNE(r_time,l_time,r_type,l_type)])
                
                if (str(s_node)+ '_' + str(l_t_node)) in self.graph_idx_all:
                    cou2 = self.graph_idx_all[str(s_node)+ '_' + str(l_t_node)]
                    b = [i for i in cou2 if i <= index]
                    if len(b) != 0:

                        r_index = random.sample(b,1)[0]
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 0
                        r_time = self.all_edges_info[r_index].time
                        motif_edges.append([l_t_node,r_time,l_time,self.find_type_MSTNE(r_time,l_time,r_type,l_type)])
            
            else:
                one_type = 3
                if l_type == -1:
                    l_type = 1
                if (str(l_s_node)+ '_' + str(s_node)) in self.graph_idx_all:
                    cou = self.graph_idx_all[str(l_s_node)+ '_' + str(s_node)]
                    b = [i for i in cou if i <= index]
                    if len(b) != 0:
                        r_index = random.sample(b,1)[0]
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 1
                        r_time = self.all_edges_info[r_index].time
                        motif_edges.append([l_t_node,r_time,l_time,self.find_type_MSTNE(r_time,l_time,r_type,l_type)])

                if (str(s_node)+ '_' + str(l_s_node)) in self.graph_idx_all:
                    cou2 = self.graph_idx_all[str(s_node)+ '_' + str(l_s_node)]
                    b = [i for i in cou2 if i <= index]
                    if len(b) != 0:
                        r_index = random.sample(b,1)[0]
                        if r_index in self.time_count_all:
                            r_type = 2
                        else:
                            r_type = 0
                        r_time = self.all_edges_info[r_index].time
                        motif_edges.append([l_t_node,r_time,l_time,self.find_type_MSTNE(r_time,l_time,r_type,l_type)])

            if len(motif_edges) == 0:
                hist_nodes.append(self.node_map[l_s_node])
                hist_time.append(l_time)
                h_t_time.append(l_time)

                if l_type == 2 and one_type <= 1:
                    type_all.append(2)
                elif l_type == 2 and one_type > 1:
                    type_all.append(5)
                else:
                    if one_type <= 1:
                        type_all.append(one_type)
                    else:
                        type_all.append(one_type+1)
                
                if type_all == 2 or type_all == 3 or type_all == 24:
                    h_t_masks[inx] = 1
                    h_l_masks[inx] = 0
                    inx += 1
                else:
                    h_t_masks[inx] = 0
                    h_l_masks[inx] = 1
                    inx += 1

                if index_2 < l_counts :
                    next_edges = self.all_edges_info[l_neighbors[index_2]]
                else:
                    next_edges = self.all_edges_info[r_neighbors[index_2-l_counts]]

                s_node = next_edges.start_node
                t_node = next_edges.end_node
                time = next_edges.time
                t_nodes.append(self.node_map[t_node])
            
            else:
                select_edges = random.randint(0,len(motif_edges)-1)
                hist_nodes.append(self.node_map[motif_edges[select_edges][0]])
                hist_time.append(motif_edges[select_edges][1])
                h_t_time.append(motif_edges[select_edges][2])

                h_t_masks[inx] = 1
                h_l_masks[inx] = 1
                inx += 1

                type_all.append(motif_edges[select_edges][3])
                if motif_edges[select_edges][1] < motif_edges[select_edges][2]:
                    s_node = s_node
                    t_node = motif_edges[select_edges][0]
                    time = motif_edges[select_edges][1]
                else:
                    s_node = motif_edges[select_edges][0]
                    t_node = t_node
                    time = motif_edges[select_edges][2]
        
        return hist_nodes,hist_time,type_all,h_t_time,h_t_masks,h_l_masks