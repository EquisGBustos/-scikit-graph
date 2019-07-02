# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:42:24 2019

@author: Fede
"""

"""
funciones implementadas:
    
degree(),
centrality(),
betweenness(),
pagerank(),
communities_greedy_modularity(),
communities_label_propagation(),
mean_neighbors(),
std_neighbors(),
max_neighbors(),
min_neighbors(),
within_module_degree(),
participation_coefficient(),
node_embeddings(),
is_RT(),
topic_Detection(),
emotion_Detection(), 
"""

#importo
import pandas as pd
import networkx as nx
import numpy as np

import grafos_funciones_paquete as gfp

#data de ejemplo
"""
G = nx.karate_club_graph() #grafo
f = pd.DataFrame(data = {'name': range(34),'col1': np.random.rand(34), 'col2': np.random.rand(34)}) #features
"""
G = nx.Graph() 
G.add_edges_from([(1, 2),(1,3),(2,3),(1,4),(5, 6),(6,7),(7,5),(5,8),(1,5)])
f = pd.DataFrame(data = {'name': range(1,9),'col1': [1,1,1,1,2,2,2,2], 'col2':[1,6,3,2,2,3,7,9]})

#nx.draw(G)

print(f.head())
f = gfp.betweenness(G,f)
f = gfp.pagerank(G,f)
f = gfp.node_embeddings(G,f,20)

print(f.head())
