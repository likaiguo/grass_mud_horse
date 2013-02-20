import networkx as nx
import matplotlib.pyplot as plt
f=file("../train/Location_Tree.csv")
a=f.readlines()
x=[]
G=nx.Graph()
for item in a:
	x.append(item[1:-2])

for item in x:
	l=item.split("~")
	edges=[]
	for node in range(0,len(l)-1):
		edges.append((l[node],l[node+1]))
	G.add_edges_from(edges)


pos=nx.graphviz_layout(G,prog='twopi',args='')
plt.figure(figsize=(8,8))
nx.draw(G,pos,node_size=20,alpha=0.5,node_color="blue", with_labels=False)
plt.axis('equal')
plt.show()
