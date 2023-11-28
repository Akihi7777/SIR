# -*- coding: utf-8 -*-
# @Author: JasonWong97
# @Date:   2018-10-19 21:37:31
# @Last Modified by:   JasonWong97
# @Last Modified time: 2018-10-21 20:45:30
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from ER_Network import ER_network as ER
from BA_Network import BA_network as BA
from SIR_Network import SIR_model as SIR
import pylab as pl
from Add_Links_By_Betweenness import Add_Links_By_Betweenness as ALBB
from Add_Links_Random import Add_Links_Random as ALR



# total_links=300
# rounds=10
# interval=10
# times=np.zeros(int(total_links/interval)+1)
# add_links=0
# x=[]
# while add_links<=total_links/interval:
#     for i in range(rounds):
#         sir_model_ba=SIR(beta=0.02,miu=0.1,t=100,network="Add_Links_By_Betweenness",method="max_node",add_links=add_links)
#         times[add_links]+=sir_model_ba.main()
#     times[add_links]=times[add_links]/rounds
#
#     x.append(add_links)
#     add_links += 1


total_links=500
rounds=1
interval=10
times=np.zeros(int(total_links/interval)+1)
add_links=0
x=[]
while add_links<=total_links/interval:
    for i in range(rounds):
        #albb=ALBB(1,1,total_links)
        albb = ALR(total_links)
        net_matrix=albb.main()
        G=nx.from_numpy_array(net_matrix)
        ave_path=nx.average_shortest_path_length(G)
        times[add_links] +=ave_path
    x.append(add_links)
    add_links += 1

times=times/rounds

x=np.array(x)
plt.plot(x*interval,times, '-k', label='Susceptibles',marker='o')
plt.legend(loc=0)
# plt.title('SIR_Model with links adding randomly')
plt.title('average_shortest_path_length')
plt.xlabel('add_links')
plt.ylabel('times')
plt.show()
#sir_model_ba=SIR(beta=0.15,miu=0.1,t=100,network="barabasi_albert_graph",method="random_set")



