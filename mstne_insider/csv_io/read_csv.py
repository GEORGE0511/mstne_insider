import datetime
import os
import csv
from tqdm import tqdm
import networkx as nx
import time
import matplotlib.pyplot as plt

def read_csv(csv_name,dir_path):
    with open(os.path.join(dir_path, csv_name), 'r') as file:
        print("..."+csv_name+"...")
        read = csv.reader(file)
        next(read)
        for i in tqdm(read):
            print(i)

def select_csv(csv_name,dir_path,start_time,end_time,output_path,row_type):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    with open(os.path.join(output_path, csv_name), 'w') as write:
        writer = csv.writer(write)
        writer.writerow(row_type)
        with open(os.path.join(dir_path, csv_name), 'r') as file:
            print("..."+csv_name+"...")
            start = datetime.datetime.strptime(start_time, '%m/%d/%Y %H:%M:%S')
            end = datetime.datetime.strptime(end_time,'%m/%d/%Y %H:%M:%S')
            read = csv.reader(file)
            next(read)
            for i in tqdm(read):
                temp = datetime.datetime.strptime(i[1], '%m/%d/%Y %H:%M:%S')
                if start <= temp and temp <= end:
                    writer.writerow(i)
    write.close()

def prase_csv(csv_name,dir_path,row_type):
    with open(os.path.join(dir_path, csv_name), 'r') as file:
        print("..."+csv_name+"...")
        #     id,date,user,pc,activity
        #     {Q4D5-W4HH44UC-5188LWZK},01/02/2010 02:24:51,JBI1134,PC-0168,Logon
        #     {G7V0-S4TP95SA-9203AOGR},01/02/2010 02:38:28,JBI1134,PC-0168,Logoff
        read = csv.reader(file)
        next(read)
        edge_list = []
        for i in tqdm(read):
            if i != []:
                i = dict(zip(row_type,i))
                timestamp = time.mktime(time.strptime(i['date'], '%m/%d/%Y %H:%M:%S'))
                i['timestamp'] = timestamp
                edge_list.append(i)
        return edge_list

def init_edge(edge_list,csv_name):
    all_edges = []
    for row in edge_list:
        att = dict(zip(node_type[csv_name]["att"], [row[j] for j in node_type[csv_name]["att"]]))
        if csv_name == 'device.csv':
            start,end = row[node_type[csv_name]["node"][0]],row[node_type[csv_name]["node"][1]].split(';')
            for end_iter in end:
                all_edges.append((start, end_iter,att))
        else:
            start,end = row[node_type[csv_name]["node"][0]],row[node_type[csv_name]["node"][1]]
            all_edges.append((start, end, att))
    return all_edges

class graph:
    def __init__(self):
        self.graph = self.init_graph()

    def init_graph(self):
        graph = nx.MultiGraph()
        return graph

    def add_edges(self,all_edges):
        self.graph.add_edges_from(all_edges)

if __name__ == '__main__':
    dir_path = r"../our_data/r_part"
    row_type = {
        "logon.csv": ["edge_id", "date", "user", "pc", "activate"],
        "device.csv": ["edge_id", "date", "user", "pc", "file_tree", "activate"],
        "http.csv": ["edge_id", "date", "user", "pc", "url","content"],
        # "email.csv": ["edge_id", "date", "user", "pc", "to", "cc","bcc","from","activity", "size", "attachments", "content"],
        "file.csv": ["edge_id", "date", "user", "pc", "filename", "activity", "to_removable_media","from_removable_media", "content","timestamp"]
    }
    node_type = {
        "logon.csv": {"node": ["user", "pc"], "att": ["activate", "timestamp","edge_id"]},
        "device.csv": {"node": ["pc", "file_tree"], "att": ["activate", "timestamp","edge_id"]},
        "http.csv": {"node": ["pc", "url"], "att": ["content", "timestamp","edge_id"]},
        # "email.csv": {"node": ["from", "to"], "att": ["content", "timestamp"]},
        # "email.csv": ["edge_id", "date", "user", "pc", "to", "cc", "bcc", "from", "activity", "size", "attachments",
        #               "content"],
        "file.csv": {"node": ["pc", "filename"], "att": ["activity", "to_removable_media","from_removable_media","content","edge_id","timestamp"]},
    }
    # read_csv("device.csv",r"../our_data/r_part")
    # for csv_name in row_type:
    #     select_csv(csv_name, r"G:\r5.1", "12/03/2010 06:05:31", "12/08/2010 23:35:42", r"../our_data/r5.1_test",
    #                row_type[csv_name])
    G = graph()
    for csv_name in row_type:
        edge_list = prase_csv(csv_name,dir_path,row_type[csv_name])
        G.add_edges(init_edge(edge_list,csv_name))
    print('number of nodes', G.graph.number_of_nodes())
    print('number of edges', G.graph.number_of_edges())
    nx.write_edgelist(G.graph, "../our_data/graph_edge_list2")
    nx.write_gpickle(G.graph, "../our_data/graph.gpickle2")



# with open(os.path.join(dir_path, "http.csv"), 'r') as file:
#     print("...logon.csv...")
#     #     id,date,user,pc,activity
#     #     {Q4D5-W4HH44UC-5188LWZK},01/02/2010 02:24:51,JBI1134,PC-0168,Logon
#     #     {G7V0-S4TP95SA-9203AOGR},01/02/2010 02:38:28,JBI1134,PC-0168,Logoff
#     read = csv.reader(file)
#     next(read)
#     start_time = datetime.datetime.strptime('08/02/2010 10:34:31','%m/%d/%Y %H:%M:%S')
#     end_time = datetime.datetime.strptime('09/30/2010 15:04:03','%m/%d/%Y %H:%M:%S')
#     for i in tqdm(read):
#         i[1] = datetime.datetime.strptime(i[1],'%m/%d/%Y %H:%M:%S')
#         if i[2] == "CCH0959" and start_time <= i[1] and i[1] <= end_time:
#             print(i)


