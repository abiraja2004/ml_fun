import networkx as nx

# create graph
G = nx.Graph()

# add risk factors
G.add_node("country")

# add edges
G.add_edge(1,2)

# print out graph
print "Nodes: ", G.nodes()
print(G.edges())
