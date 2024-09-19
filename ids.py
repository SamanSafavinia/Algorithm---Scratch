#%%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def bi_IDS(graph):
    color = {}

    def is_bipartite(node, depth, c):
        color[node] = c
        if depth == 0:
            return True
        for neighbor in graph[node]:
            if neighbor in color:
                if color[neighbor] == c:
                    return False
            else:
                if not is_bipartite(neighbor, depth - 1, 1 - c):
                    return False
        return True

    # Iterative Part
    for depth in range(len(graph)):
        color = {}
        for node in graph:
            if node not in color:
                if not is_bipartite(node, depth, 0):
                    return False
    return True

def bi_BFS(graph):
    colors = {}
    queue = []

    for node in graph.nodes():
        if node not in colors:
            colors[node] = 1
            queue.append(node)

            while queue:
                current = queue.pop(0)

                for neighbor in graph.neighbors(current):
                    if neighbor not in colors:
                        colors[neighbor] = 1 - colors[current]
                        queue.append(neighbor)
                    elif colors[neighbor] == colors[current]:
                        return False

    return True

def bi_DFS(graph):
    colors = {}
    stack = []

    for node in graph.nodes():
        if node not in colors:
            colors[node] = 1
            stack.append(node)

            while stack:
                current = stack.pop()

                for neighbor in graph.neighbors(current):
                    if neighbor not in colors:
                        colors[neighbor] = 1 - colors[current]
                        stack.append(neighbor)
                    elif colors[neighbor] == colors[current]:
                        return False

    return True



#%%
# My Examples
g = nx.Graph()
g.add_nodes_from([1,2,3,4,5])
g.add_edge(1,2)
g.add_edge(2,1)
g.add_edge(3,1)

g.add_edge(5,1)
g.add_edge(5,3)
g.add_edge(3,4)
g.add_edge(4,2)

nx.draw(g,with_labels=True)
plt.draw()
plt.show()

print("BFS Bipartite:", bi_BFS(g))
print("DFS Bipartite:", bi_DFS(g))
print("DFS Bipartite:", bi_IDS(g))


#%%

#Proj Examples 
g = nx.read_edgelist(r'C:\Users\Saman\Desktop\Project 1 - AI\graph3.edgelist')
nx.draw(g,with_labels=True)
plt.draw()
plt.show()

print("BFS Bipartite:", bi_BFS(g))
print("DFS Bipartite:", bi_DFS(g))
print("DFS Bipartite:", bi_IDS(g))


#%%
# Coloring
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

def color_bipartite(graph):
    def bfs(node):
        colors[node] = 0
        queue = deque([node])

        while queue:
            current = queue.popleft()

            for neighbor in graph.neighbors(current):
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[current]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[current]:
                    return False
        return True

    colors = {}
    for node in graph.nodes():
        if node not in colors:
            if not bfs(node):
                return "Graph is not bipartite"

    color_map = ['red' if colors[node] == 0 else 'blue' for node in graph.nodes()]

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_color=color_map, with_labels=True)
    plt.show()

    return "Bipartite"

print("BFS Bipartite:", color_bipartite(g))

g = nx.Graph()
g.add_nodes_from([1,2,3,4,5])
g.add_edge(1,2)
g.add_edge(2,1)
g.add_edge(3,1)

#g.add_edge(5,1)
#g.add_edge(5,3)
g.add_edge(3,4)
g.add_edge(4,2)

nx.draw(g,with_labels=True)
plt.draw()
plt.show()
print("BFS Bipartite:", color_bipartite(g))













# %%
