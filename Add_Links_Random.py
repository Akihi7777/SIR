import random
import numpy as np
import networkx as nx
import community.community_louvain as cl
import operator

class Add_Links_Random:
    def __init__(self,k):
        self.net_matrix = np.loadtxt('data.txt', delimiter=' ')
        self.G = nx.from_numpy_array(self.net_matrix)
        self.k = k

    def community_lovain(self):
        self.partition = cl.best_partition(self.G)
        self.community_graph = []
        for com in set(self.partition.values()):
            list_nodes = [nodes for nodes in self.partition.keys() if self.partition[nodes] == com]
            self.community_graph.append(nx.subgraph(self.G, list_nodes))

    def add_links(self):
        nodes=list(self.G.nodes)
        for i in range(self.k):
            node=random.choice(nodes)
            nodes_list_copy=nodes.copy()
            nodes_list_copy.remove(node)
            random.shuffle(nodes_list_copy)
            for node_candidate in nodes_list_copy:
                if self.G.has_edge(node,node_candidate)==False and self.partition[node]!=self.partition[node_candidate]:
                    self.G.add_edge(node,node_candidate)
                    break
        #print('count:',self.edge_count)
        return nx.to_numpy_array(self.G)

    def main(self):
        self.community_lovain()
        G_matrix=self.add_links()
        return G_matrix
# if __name__ == '__main__':
#     a=Add_Links_Random(100)
#     a.community_lovain()
#     a.add_links()
