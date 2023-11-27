import random
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from ER_Network import ER_network as ER
from BA_Network import BA_network as BA
import networkx as nx
import community.community_louvain as cl
import operator

random.seed(2)
class Add_Links_By_Betweenness:
    def __init__(self,n,m):
        self.G=nx.barabasi_albert_graph(n, m)
        self.net_matrix=nx.to_numpy_array(self.G)


    def community_lovain(self):
        self.partition = cl.best_partition(self.G)
        self.community_graph=[]
        for com in set(self.partition.values()):
            list_nodes = [nodes for nodes in self.partition.keys() if self.partition[nodes] == com]
            self.community_graph.append(nx.subgraph(self.G,list_nodes))

    def betweenness_without_decay(self,k=0.3):
        edge_count=0
        node_num=int(k*len(self.G.nodes))
        G_betweenness = nx.betweenness_centrality(self.G)
        high_betweenness_nodes = sorted(G_betweenness.items(), key=operator.itemgetter(1), reverse=True)[:node_num]
        high_betweenness_nodes=[item[0] for item in high_betweenness_nodes]
        for node in high_betweenness_nodes:
            nodes_list_copy=high_betweenness_nodes.copy()
            nodes_list_copy.remove(node)
            random.shuffle(nodes_list_copy)
            for node_candidate in nodes_list_copy:
                if self.G.has_edge(node,node_candidate)==False and self.partition[node]!=self.partition[node_candidate]:
                    self.G.add_edge(node,node_candidate)
                    edge_count+=1
                    break
        print('count:',edge_count)

if __name__ == '__main__':
    a=Add_Links_By_Betweenness(500,3)
    a.community_lovain()
    a.betweenness_without_decay()