import random
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from ER_Network import ER_network as ER
from BA_Network import BA_network as BA
from Add_Links_By_Betweenness import Add_Links_By_Betweenness as ALBB
import networkx as nx

G=nx.barabasi_albert_graph(500, 3)
net_matrix=nx.to_numpy_array(G)
np.savetxt('data1.txt',net_matrix)
# data=np.loadtxt('data.txt',delimiter=' ')
# print(len(data))