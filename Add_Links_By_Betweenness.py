import random
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from ER_Network import ER_network as ER
from BA_Network import BA_network as BA
import networkx as nx
import community.community_louvain as cl

random.seed(2)
class Add_Links_By_Betweenness:
    def __init__(self,n,m):
        self.G=nx.barabasi_albert_graph(n, m)
        self.net_matrix=nx.to_numpy_array(self.G)


    def community_lovain(self):
        self.partition = cl.best_partition(self.G)
        self.community_list=dict()
        for com in set(self.partition.values()):
            list_nodes = [nodes for nodes in self.partition.keys() if self.partition[nodes] == com]
            self.community_list[com]=list_nodes

    def betweenness_with_decay(self):


if __name__ == '__main__':
    a=Add_Links_By_Betweenness(500,3)
    a.community_lovain()
    print(a.community_list)