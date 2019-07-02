#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:25:16 2019

@author: usuario

ejemplo del pipline y por que sirve:
    https://dreisbach.us/articles/building-scikit-learn-compatible-transformers/
    
otras formas de hacerlo:
    https://towardsdatascience.com/a-simple-example-of-pipeline-in-machine-learning-with-scikit-learn-e726ffbb6976
    https://github.com/suvoooo/Machine_Learning/blob/master/pipelineWine.py

Ayuda:
    https://stackoverflow.com/questions/50965004/sklearn-custom-transformers-difference-between-using-functiontransformer-and-su
    
"""


#importo
import pandas as pd
import networkx as nx
import numpy as np

import grafos_funciones_paquete as gfp #tiene funciones
import grafos_funciones_paquete_V2 as gfp2 #tiene transformers


#data de ejemplo
num_nodos = 200
num_aristas = 350
G = nx.gnm_random_graph(num_nodos , num_aristas) 
f = pd.DataFrame(data = {'name': range(num_nodos),
                         'col1': np.random.rand(num_nodos), 
                         'col2': np.random.rand(num_nodos) > 0.5})

#nx.draw(G)

print(f.head())
f = gfp.betweenness(G,f)
f = gfp.pagerank(G,f)
f = gfp.communities_label_propagation(G,f)
print(f.head())
f["target"] = [1 if i >0.005 else 0  for i in f.pagerank]

from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV


X=f.drop(['target'],axis=1)
Y=f['target']

#++++++++++++++++++++++++++++++++
# create the pipeline object
#++++++++++++++++++++++++++++++++
Dumb = gfp2.Dumb(5)
Replace1 = gfp2.Replace(np.nan,-1)
Replace2 = gfp2.Replace(np.inf,-1)
Degree = gfp2.Degree(G)
DropName = gfp2.DropName()
Betweenness = gfp2.Betweenness(G)
Pagerank = gfp2.Pagerank(G)
Clustering = gfp2.Clustering(G)
Centrality = gfp2.Centrality(G)
Communities_label_propagation = gfp2.Communities_label_propagation(G)
Communities_greedy_modularity = gfp2.Communities_greedy_modularity(G)
Mean_neighbors = gfp2.Mean_neighbors(G, "col1", 1)
Std_neighbors = gfp2.Std_neighbors(G, "col1", 1)
Max_neighbors = gfp2.Max_neighbors(G, "col1", 1)
Min_neighbors = gfp2.Min_neighbors(G, "col1", 1)
Participation_coefficient = gfp2.Participation_coefficient(G, "communities_label_propagation")
Within_module_degree = gfp2.Within_module_degree(G, "communities_label_propagation")
Node_embeddings = gfp2.Node_embeddings(G)


def aux_f1(G):
    return [0 for i in G.nodes()]

def aux_f2(G, X):
    return [0 for i in G.nodes()]

Graph_fuction = gfp2.Graph_fuction(G, aux_f1)
Graph_features_fuction = gfp2.Graph_features_fuction(G, aux_f2)

steps = [#("nioqui", Dumb), ("Degree", Degree), ("Degree2", Degree), 
         #("Betweenness", Betweenness), ("Pagerank", Pagerank), ("Centrality", Centrality), 
         #("CLP", Communities_label_propagation), ("CGM", Communities_greedy_modularity),
         #("mean", Mean_neighbors), ("std", Std_neighbors), ("max", Max_neighbors),  ("min", Min_neighbors), 
         #("PC", Participation_coefficient),  ("WMD", Within_module_degree), 
         #("embeddings", Node_embeddings), 
         #("Graph_fuction", Graph_fuction), ("Graph_features_fuction", Graph_features_fuction)
         ("Clustering", Clustering),
         ("DropName", DropName), ("FillNans", Replace1), ("FillInf", Replace2), 
         ('scaler', StandardScaler()), ('SVM', SVC())]

pipeline = Pipeline(steps)


#++++++++++++++++++++++++++++++++++++++
#+ create the hyperparameter space
#++++++++++++++++++++++++++++++++++++++

parameteres = {'SVM__C':[0.001,0.1,10,100,10e5], 'SVM__gamma':[0.1,0.01]}

#++++++++++++++++++++++++++++++++++++
#+ create train and test sets
#++++++++++++++++++++++++++++++++++++

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=30, stratify=Y)

#print X_test.shape

#++++++++++++++++++++++++++++++
#+ Grid Search Cross Validation
#++++++++++++++++++++++++++++++
grid = RandomizedSearchCV(pipeline, param_distributions = parameteres, cv=5, n_iter = 2)

grid.fit(X_train, y_train)

print('score: ', grid.score(X_test,y_test)) 


















