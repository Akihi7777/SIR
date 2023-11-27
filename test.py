import networkx as nx

# 创建原始图
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])

# 定义特定节点集合 U
U = {1,2,  4}

# 获取子图
subgraph = G.subgraph(U)

# 打印子图的节点和边
print("Subgraph Nodes:", subgraph.nodes())
print("Subgraph Edges:", subgraph.edges())