
import networkx as nx
import matplotlib.pyplot as plt

import community.community_louvain as cl
import pylab as pl
from ER_Network import ER_network as ER
from BA_Network import BA_network as BA

ba_network=BA(N=3,p=0.006,N_end=1000,m0=3,title="BA network")
net_matrix=ba_network.Create_BA_network(ba_network.Create_ER_network())

G=nx.from_numpy_array(net_matrix)
partition = cl.best_partition(G)

# drawing
size = float(len(set(partition.values())))
print(len(set(partition.values())))
pos = nx.shell_layout(G)
count = 0.
for com in set(partition.values()):
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
    print(list_nodes)
    nx.draw_networkx_nodes(G, pos, list_nodes,node_color=str(count / size))

nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()