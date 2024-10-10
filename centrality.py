import cxrandomwalk as rw
import igraph as ig
import numpy as np
g = ig.Graph.Erdos_Renyi(n=10,m=30)
weights = np.random.random(size=g.ecount())
g.es["weight"] = weights
x1 = g.betweenness(weights="weight")
agent = rw.Agent(g.vcount(),g.get_edgelist(),False,weights)
x = agent.betweenness()

print(x1)
print(x)