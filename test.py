import random

import networkx as nx

random.seed(1)
# 创建原始图
l=[1,2,3,4,5,6]
l_c=l.copy()
l_c.remove(1)
for i in range(5):
    random.shuffle(l_c)
    print(l)
    print(l_c)