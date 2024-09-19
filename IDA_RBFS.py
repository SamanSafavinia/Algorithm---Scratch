#!/usr/bin/env python
# coding: utf-8

# In[6]:


import networkx as nx
import sys
import matplotlib.pyplot as plt


# In[7]:


graph = {
    'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucharest': 90},
    'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}


# In[8]:


heuristic = {
    'Arad': 366,
    'Zerind': 374,
    'Oradea': 380,
    'Sibiu': 253,
    'Timisoara': 329,
    'Lugoj': 244,
    'Mehadia': 241,
    'Drobeta': 242,
    'Craiova': 160,
    'Rimnicu Vilcea': 193,
    'Fagaras': 176,
    'Pitesti': 100,
    'Bucharest': 0,
    'Giurgiu': 77,
    'Urziceni': 80,
    'Hirsova': 151,
    'Eforie': 161,
    'Vaslui': 199,
    'Iasi': 226,
    'Neamt': 234
}


# In[22]:


G = nx.Graph()


for city, connections in graph.items():
    for neighbor, cost in connections.items():
        G.add_edge(city, neighbor, weight=cost)


nx.set_node_attributes(G, heuristic, 'heuristic')

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
node_colors = [heuristic[node] for node in G.nodes()]

nx.draw(G, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.plasma, node_size=800, font_weight='bold')

labels = nx.get_node_attributes(G,G)
nx.draw_networkx_labels(G, pos, labels=labels)



edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title('Romania Cities')
plt.show()


# In[10]:


import sys

def IDA_star(graph, heuristic, start, goal):
    def search(path, g, threshold):
        node = path[-1]
        f = g + heuristic[node]
        if f > threshold:
            return f
        if node == goal:
            return True
        min_val = sys.maxsize
        for succ in graph[node]:
            if succ not in path:
                path.append(succ)
                t = search(path, g + graph[node][succ], threshold)
                if t == True:
                    return True
                if t < sys.maxsize:
                    min_val = min(min_val, t)
                path.pop()
        return min_val

    threshold = heuristic[start]
    path = [start]
    while True:
        t = search(path, 0, threshold)
        if t == True:
            return path
        if t == sys.maxsize:
            return None
        threshold = t

    


# In[14]:


start_city = "Craiova"
print(IDA_star(graph, heuristic, start_city, "Bucharest"))



import heapq

def RBFS(graph, heuristic, start, goal):
    fringe = [(0, start, [start])]
    visited = set()

    while fringe:
        cost, node, path = heapq.heappop(fringe)

        if node == goal:
            return cost, path 

        visited.add(node)

        for neighbor in graph[node]:
            if neighbor in visited:
                continue

            f_value = cost + graph[node][neighbor] + heuristic[neighbor]
            new_path = path + [neighbor]

            heapq.heappush(fringe, (f_value, neighbor, new_path))

    return -1, [] 


graph = {
    'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucharest': 90},
    'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87}
}


heuristic = {
    'Arad': 366,
    'Zerind': 374,
    'Oradea': 380,
    'Sibiu': 253,
    'Timisoara': 329,
    'Lugoj': 244,
    'Mehadia': 241,
    'Drobeta': 242,
    'Craiova': 160,
    'Rimnicu Vilcea': 193,
    'Fagaras': 176,
    'Pitesti': 100,
    'Bucharest': 0,
    'Giurgiu': 77,
    'Urziceni': 80,
    'Hirsova': 151,
    'Eforie': 161,
    'Vaslui': 199,
    'Iasi': 226,
    'Neamt': 234
}

start = 'Craiova'
goal = 'Bucharest'

result_cost, result_path = RBFS(graph, heuristic, start, goal)

if result_cost != -1:
    print("Cost:", result_cost)
    print("Optimal Path:", ' -> '.join(result_path))
else:
    print("No path found.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




