#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:42:50 2019

@author: usuario
"""


import pandas as pd
import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


#polemico esto:
import warnings
warnings.filterwarnings("ignore")

"""
lo mismo que la version 1, pero ahora con transformadores.
funciones implementadas:
    
degree(),


Falta:
    agregar cosas que chequeen errores (tipo si no todoslos nodos estan en el dataframe, etc.)

"""

#Transformers
class Dumb(BaseEstimator, TransformerMixin):
    def __init__(self,m = 8):
        self.m = m
        print('a',self.m)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print('b',self.m)
        return X
    
class Replace(BaseEstimator, TransformerMixin):
    def __init__(self, value1,value2):
        self.value1 = value1
        self.value2 = value2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.replace(self.value1, self.value2, regex=True)    
    
class DropName(BaseEstimator, TransformerMixin):
    """
    Drops the "name" column.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_prima = X.drop(['name'],axis=1)
        return X_prima    

class Graph_fuction(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the degree of each node.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    function: a python function that takes the graph G as input and outpus a column of the same length that the number of nodes in the graph.
    column_name: a string with the name of the column
    """
    def __init__(self, G, function, column_name = "Graph_fuction"):
        self.G = G
        self.function = function
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) #por si hay cross validation, me quedo con el subset correcto
        column =  self.function(G_train)
        degree_df = pd.DataFrame(data = {'name': list(G_train.nodes()), self.column_name: column })  
        X_prima = pd.merge(X, degree_df, on='name')
        print(X_prima.columns)
        return X_prima

class Graph_features_fuction(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the degree of each node.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    function: a python function that takes the graph G as input and outpus a column of the same length that the number of nodes in the graph.
    column_name: a string with the name of the column
    """
    def __init__(self, G, function, column_name = "Graph_features_fuction"):
        self.G = G
        self.function = function
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) #por si hay cross validation, me quedo con el subset correcto
        column =  self.function(G_train, X)
        degree_df = pd.DataFrame(data = {'name': list(G_train.nodes()), self.column_name: column })  
        X_prima = pd.merge(X, degree_df, on='name')
        print(X_prima.columns)
        return X_prima
    
class Degree(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the degree of each node.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    """
    def __init__(self, G):
        self.G = G

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) #por si hay cross validation, me quedo con el subset correcto
        degree_dic =  nx.degree_centrality(G_train)
        degree_df = pd.DataFrame(data = {'name': list(degree_dic.keys()), 'degree': list(degree_dic.values()) })  
        X_prima = pd.merge(X, degree_df, on='name')
        return X_prima
    
class Clustering(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the degree of each node.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    """
    def __init__(self, G):
        self.G = G

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) #por si hay cross validation, me quedo con el subset correcto
        clustering_dic = nx.clustering(G_train)
        clustering_df = pd.DataFrame(data = {'name': list(clustering_dic.keys()), 'clustering': list(clustering_dic.values()) })  
        X_prima = pd.merge(X, clustering_df, on='name')
        return X_prima    
    
class Centrality(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the centrality of each node.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    """
    def __init__(self, G):
        self.G = G

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) #por si hay cross validation, me quedo con el subset correcto
        centrality_dic = nx.degree_centrality(G_train)
        centrality_df = pd.DataFrame(data = {'name': list(centrality_dic.keys()), 'centrality': list(centrality_dic.values()) })  
        X_prima = pd.merge(X, centrality_df, on='name')
        return X_prima    

class Betweenness(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the betweenness of each node.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    """
    def __init__(self, G):
        self.G = G

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) #por si hay cross validation, me quedo con el subset correcto
        betweenness_dic = nx.betweenness_centrality(G_train)
        betweenness_df = pd.DataFrame(data = {'name': list(betweenness_dic.keys()), 'betweenness': list(betweenness_dic.values()) })  
        X_prima = pd.merge(X, betweenness_df, on='name')
        return X_prima    

class Pagerank(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the pagerank of each node.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    """
    def __init__(self, G):
        self.G = G

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) #por si hay cross validation, me quedo con el subset correcto
        pagerank_dic = nx.pagerank(G_train)
        pagerank_df = pd.DataFrame(data = {'name': list(pagerank_dic.keys()), 'pagerank': list(pagerank_dic.values()) })  
        X_prima = pd.merge(X, pagerank_df, on='name')
        return X_prima

    
class Communities_greedy_modularity(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the comunity of each node.
    The comunitys are detected using greedy modularity.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    """    
    # funciona con la version de '2.4rc1.dev_20190610203526' de netwrokx (no con la 2.1)
    def __init__(self, G):
        self.G = G

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) #por si hay cross validation, me quedo con el subset correcto
        communities_dic = nx.algorithms.community.greedy_modularity_communities(G_train)
        communities_df = pd.DataFrame(data = {'name': [i for j in range(len(communities_dic)) for i in list(communities_dic[j])], 'communities_greedy_modularity': [j for j in range(len(communities_dic)) for i in list(communities_dic[j])] })  
        X_prima = pd.merge(X,communities_df, on='name')
        return X_prima    
    
    

class Communities_label_propagation(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the comunity of each node.
    The comunitys are detected using glabel propagation.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    """    
    # funciona con la version de '2.4rc1.dev_20190610203526' de netwrokx (no con la 2.1)
    def __init__(self, G):
        self.G = G

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) #por si hay cross validation, me quedo con el subset correcto
        communities_gen = nx.algorithms.community.label_propagation_communities(G_train)
        communities_dic = [comunidad for comunidad in communities_gen]
        communities_df = pd.DataFrame(data = {'name': [i for j in range(len(communities_dic)) for i in list(communities_dic[j])], 'communities_greedy_modularity': [j for j in range(len(communities_dic)) for i in list(communities_dic[j])] })  
        X_prima = pd.merge(X,communities_df, on='name')
        return X_prima    
    

class Mean_neighbors(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the mean value of its neigbors feature.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    column: the column to which the mean is applied.
    n: no se como decirlo: seria a primeros vecinos o segundos vecinos.
    """
    def __init__(self, G, column, n=1):
        self.G = G
        self.column = column
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) #por si hay cross validation, me quedo con el subset correcto
        mean_neighbors = np.zeros([X.shape[0]])
        matrix = nx.to_numpy_matrix(G_train)
        for e in range(1,self.n):
            matrix += matrix ** e
        for i in range(X.shape[0]):
            neighbors = matrix[i]>0
            mean_neighbors[i] = X[neighbors.tolist()[0]][self.column].mean() 
        X_prima = X
        X_prima["mean_" + str(self.n) + "_neighbors_" + str(self.column)] = mean_neighbors
        return X_prima
    
    
class Std_neighbors(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the standar desviation value of its neigbors feature.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    column: the column to which the mean is applied.
    n: no se como decirlo: seria a primeros vecinos o segundos vecinos.
    """
    def __init__(self, G, column, n=1):
        self.G = G
        self.column = column
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) #por si hay cross validation, me quedo con el subset correcto
        
        std_neighbors = np.zeros([X.shape[0]])
        matrix = nx.to_numpy_matrix(G_train)
        for e in range(1,self.n):
            matrix += matrix ** e
        for i in range(X.shape[0]):
            neighbors = matrix[i]>0
            std_neighbors[i] = X[neighbors.tolist()[0]][self.column].std()
        X_prima = X
        X_prima["std_" + str(self.n) + "_neighbors_" + str(self.column)] = std_neighbors
        return X_prima    
    
    
class Max_neighbors(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the maximum value of its neigbors feature.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    column: the column to which the mean is applied.
    n: no se como decirlo: seria a primeros vecinos o segundos vecinos.
    """
    def __init__(self, G, column, n=1):
        self.G = G
        self.column = column
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) #por si hay cross validation, me quedo con el subset correcto
        
        max_neighbors = np.zeros([X.shape[0]])
        matrix = nx.to_numpy_matrix(G_train)
        for e in range(1,self.n):
            matrix += matrix ** e
        for i in range(X.shape[0]):
            neighbors = matrix[i]>0
            max_neighbors[i] = X[neighbors.tolist()[0]][self.column].max()
        X_prima = X
        X_prima["max_" + str(self.n) + "_neighbors_" + str(self.column)] = max_neighbors
        return X_prima       

class Min_neighbors(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the minimum value of its neigbors feature.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    column: the column to which the mean is applied.
    n: no se como decirlo: seria a primeros vecinos o segundos vecinos.
    """
    def __init__(self, G, column, n=1):
        self.G = G
        self.column = column
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) #por si hay cross validation, me quedo con el subset correcto
        
        min_neighbors = np.zeros([X.shape[0]])
        matrix = nx.to_numpy_matrix(G_train)
        for e in range(1,self.n):
            matrix += matrix ** e
        for i in range(X.shape[0]):
            neighbors = matrix[i]>0
            min_neighbors[i] = X[neighbors.tolist()[0]][self.column].min()
        X_prima = X
        X_prima["min_" + str(self.n) + "_neighbors_" + str(self.column)] = min_neighbors
        return X_prima          
    

    
class Within_module_degree(BaseEstimator, TransformerMixin):
    """
    within_module_degree calculates: Zi = (ki-ks)/Ss 
    Ki = # de links entre un nodo i y todos los de su propio cluster Si
    Ks = promedio de links de los nodos del cluster s
    Ss = std de los links de los nodos del cluster s

    The within-module degree z-score measures how well-connected node i is to other nodes in the module.

    PAPER: Guimera, R., & Amaral, L. A. N. (2005). Functional cartography of complex metabolic networks. nature, 433(7028), 895.
    
    G: a networkx graph.
    columna_comunidades: a column of the dataframe with the comunities for each node. If None, the comunities will be estimated using metodo comunidades.
    metodo_comunidades: method to calculate the communities in the graph G if they are not provided with columna_comunidades. 
    """
    def __init__(self, G, columna_comunidades = None, metodo_comunidades = "label_propagation"):
        self.G = G
        self.columna_comunidades = columna_comunidades

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) #por si hay cross validation, me quedo con el subset correcto 
        z_df = pd.DataFrame(data = {'name': [], 'within_module_degree': [] }) 
        for comunidad in set(X[self.columna_comunidades]):
            G2 = G_train.subgraph(X[X[self.columna_comunidades] == comunidad]["name"].values)
            Ks = 2*len(G2.edges) / len(G2.nodes)
            Ss = np.std([i[1] for i in G2.degree()])
            z_df = pd.concat([z_df,pd.DataFrame(data = {'name': list(G2.nodes), 'within_module_degree': [np.divide(i[1]-Ks, Ss) for i in G2.degree()] }) ])
        
        X_prima = pd.merge(X, z_df, on='name')
        return X_prima


 
class Participation_coefficient(BaseEstimator, TransformerMixin):
    """
    participation_coefficient calculates: Pi = 1- sum_s( (Kis/Kit)^2 ) 
    Kis = # de links entre el nodo i y los nodos del cluster s
    Kit = grado total del nodo i

    The participation coefficient of a node is therefore close to 1 if its links are uniformly distributed among all the modules and 0 if all its links are within its own module.
    
    PAPER: Guimera, R., & Amaral, L. A. N. (2005). Functional cartography of complex metabolic networks. nature, 433(7028), 895.
    
    G: a networkx graph.
    f: a pandas dataframe.
    columna_comunidades: a column of the dataframe with the comunities for each node. If None, the comunities will be estimated using metodo comunidades.
    metodo_comunidades: method to calculate the communities in the graph G if they are not provided with columna_comunidades. 
    """
    def __init__(self, G, columna_comunidades):
        self.G = G
        self.columna_comunidades = columna_comunidades

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) #por si hay cross validation, me quedo con el subset correcto   
        p_df = pd.DataFrame(data = {'name': X['name'], 'participation_coefficient': [1 for _ in X['name']] }) 
        for nodo in X['name']:
            Kit = len(G_train.edges(nodo))
            for comunidad in set(X[self.columna_comunidades]): 
                Kis = len([edge for edge in G_train.edges(nodo) if edge[1] in X[ X[self.columna_comunidades] == comunidad ]["name"]])
                p_df.loc[ p_df["name"] == nodo, 'participation_coefficient' ] -= np.divide(Kis, Kit) ** 2     
        X_prima = pd.merge(X, p_df, on='name')  
        return X_prima
 
class Node_embeddings(BaseEstimator, TransformerMixin):
    """
    Adds the embeddings of the nodes to the dataframe f.
    G: a networkx graph.
    f: a pandas dataframe.
    dim: the dimension of the embedding.
    """
    #https://towardsdatascience.com/node2vec-embeddings-for-graph-data-32a866340fef
    #https://github.com/eliorc/Medium/blob/master/Nod2Vec-FIFA17-Example.ipynb
    #funciona con node2vec
    def __init__(self, G,dim=20, walk_length=16, num_walks=100, workers=2):
        self.G = G
        self.dim = dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from node2vec import Node2Vec
        node2vec = Node2Vec(self.G, dimensions=self.dim, walk_length=self.walk_length, num_walks=self.num_walks, workers=self.workers)
        model = node2vec.fit(window=10, min_count=1)
    
        embeddings_df = pd.DataFrame(columns = ['name']+['node_embeddings_'+str(i) for i in range(self.dim)])
        embeddings_df['name'] = X['name']
        for name in embeddings_df['name']:
            embeddings_df[embeddings_df['name'] == name] = [name] + list(model[str(name)])
        X_prima = pd.merge(X, embeddings_df, on='name')
        return X_prima        
