#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:09:59 2019

@author: usuario
"""

import pandas as pd
import networkx as nx
import numpy as np
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

#funciones
def degree(G,f):
    """
    Adds a column to the dataframe f with the degree of each node.
    G: a networkx graph.
    f: a pandas dataframe.
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('Los tamaños del grafo y del dataframe no son inguales')   
    degree_dic =  nx.degree_centrality(G)
    degree_df = pd.DataFrame(data = {'name': list(degree_dic.keys()), 'degree': list(degree_dic.values()) })  
    f = pd.merge(f, degree_df, on='name')
    return f

def centrality(G,f):
    """
    Adds a column to the dataframe f with the centrality of each node.
    G: a networkx graph.
    f: a pandas dataframe.
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('Los tamaños del grafo y del dataframe no son inguales')   
    centrality_dic = nx.degree_centrality(G)
    centrality_df = pd.DataFrame(data = {'name': list(centrality_dic.keys()), 'centrality': list(centrality_dic.values()) })  
    f = pd.merge(f, centrality_df, on='name')
    return f

def betweenness(G,f):
    """
    Adds a column to the dataframe f with the betweenness of each node.
    G: a networkx graph.
    f: a pandas dataframe.
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('Los tamaños del grafo y del dataframe no son inguales')   
    betweenness_dic = nx.betweenness_centrality(G)
    betweenness_df = pd.DataFrame(data = {'name': list(betweenness_dic.keys()), 'betweenness': list(betweenness_dic.values()) })  
    f = pd.merge(f, betweenness_df, on='name')
    return f

def pagerank(G,f):
    """
    Adds a column to the dataframe f with the pagerank of each node.
    G: a networkx graph.
    f: a pandas dataframe.
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('Los tamaños del grafo y del dataframe no son inguales')   
    pagerank_dic = nx.pagerank(G)
    pagerank_df = pd.DataFrame(data = {'name': list(pagerank_dic.keys()), 'pagerank': list(pagerank_dic.values()) })  
    f = pd.merge(f, pagerank_df, on='name')
    return f

def clustering(G,f):
    """
    Adds a column to the dataframe f with the clustering coeficient of each node.
    G: a networkx graph.
    f: a pandas dataframe.
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('Los tamaños del grafo y del dataframe no son inguales')   
    clustering_dic = nx.clustering(G)
    clustering_df = pd.DataFrame(data = {'name': list(clustering_dic.keys()), 'clustering': list(clustering_dic.values()) })  
    f = pd.merge(f, clustering_df, on='name')
    return f

def communities_greedy_modularity(G,f):
    """
    Adds a column to the dataframe f with the comunity of each node.
    The comunitys are detected using greedy modularity.
    G: a networkx graph.
    f: a pandas dataframe.
    """
    # funciona con la version de '2.4rc1.dev_20190610203526' de netwrokx (no con la 2.1)
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('Los tamaños del grafo y del dataframe no son inguales')   
    communities_dic = nx.algorithms.community.greedy_modularity_communities(G)
    communities_df = pd.DataFrame(data = {'name': [i for j in range(len(communities_dic)) for i in list(communities_dic[j])], 'communities_greedy_modularity': [j for j in range(len(communities_dic)) for i in list(communities_dic[j])] })  
    f = pd.merge(f, communities_df, on='name')
    return f

def communities_label_propagation(G,f):
    """
    Adds a column to the dataframe f with the comunity of each node.
    The comunitys are detected using glabel propagation.
    G: a networkx graph.
    f: a pandas dataframe.
    """
    # funciona con la version de '2.4rc1.dev_20190610203526' de netwrokx (no con la 2.1)
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('Los tamaños del grafo y del dataframe no son inguales')   
    communities_gen = nx.algorithms.community.label_propagation_communities(G)
    communities_dic = [comunidad for comunidad in communities_gen]
    communities_df = pd.DataFrame(data = {'name': [i for j in range(len(communities_dic)) for i in list(communities_dic[j])], 'communities_label_propagation': [j for j in range(len(communities_dic)) for i in list(communities_dic[j])] })  
    f = pd.merge(f, communities_df, on='name')
    return f

def mean_neighbors(G,f,column,n=1):
    """
    Adds a column to the dataframe f with the mean value of its neigbors feature.
    G: a networkx graph.
    f: a pandas dataframe.
    column: the column to which the mean is applied.
    n: no se como decirlo: seria a primeros vecinos o segundos vecinos.
    """
    #podria mejorarse con matrices esparsas
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('Los tamaños del grafo y del dataframe no son inguales')   
    mean_neighbors = np.zeros([f.shape[0]])
    matrix = nx.to_numpy_matrix(G)
    for e in range(1,n):
        matrix += matrix ** e
    for i in f.index:
        neighbors = matrix[i]>0
        mean_neighbors[i] = f[neighbors.tolist()[0]][column].mean()
    f["mean_neighbors"] = mean_neighbors
    return f

def std_neighbors(G,f,column,n=1):
    """
    Adds a column to the dataframe f with the standar desviation value of its neigbors feature.
    G: a networkx graph.
    f: a pandas dataframe.
    column: the column to which the mean is applied.
    n: no se como decirlo: seria a primeros vecinos o segundos vecinos.
    """
    #podria mejorarse con matrices esparsas
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('Los tamaños del grafo y del dataframe no son inguales')   
    std_neighbors = np.zeros([f.shape[0]])
    matrix = nx.to_numpy_matrix(G)
    for e in range(1,n):
        matrix += matrix ** e
    for i in f.index:
        neighbors = matrix[i]>0
        std_neighbors[i] = f[neighbors.tolist()[0]][column].std()
    f["std_neighbors"] = std_neighbors
    return f

def max_neighbors(G,f,column,n=1):
    """
    Adds a column to the dataframe f with the maximum value of its neigbors feature.
    G: a networkx graph.
    f: a pandas dataframe.
    column: the column to which the mean is applied.
    n: no se como decirlo: seria a primeros vecinos o segundos vecinos.
    """
    #podria mejorarse con matrices esparsas
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('Los tamaños del grafo y del dataframe no son inguales')   
    max_neighbors = np.zeros([f.shape[0]])
    matrix = nx.to_numpy_matrix(G)
    for e in range(1,n):
        matrix += matrix ** e
    for i in f.index:
        neighbors = matrix[i]>0
        max_neighbors[i] = f[neighbors.tolist()[0]][column].max()
    f["max_neighbors"] = max_neighbors
    return f

def min_neighbors(G,f,column,n=1):
    """
    Adds a column to the dataframe f with the minimum value of its neigbors feature.
    G: a networkx graph.
    f: a pandas dataframe.
    column: the column to which the mean is applied.
    n: no se como decirlo: seria a primeros vecinos o segundos vecinos.
    """
    #podria mejorarse con matrices esparsas
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('Los tamaños del grafo y del dataframe no son inguales')   
    min_neighbors = np.zeros([f.shape[0]])
    matrix = nx.to_numpy_matrix(G)
    for e in range(1,n):
        matrix += matrix ** e
    for i in f.index:
        neighbors = matrix[i]>0
        min_neighbors[i] = f[neighbors.tolist()[0]][column].min()
    f["min_neighbors"] = min_neighbors
    return f

def within_module_degree(G,f, columna_comunidades = None, metodo_comunidades = "label_propagation"):
    """ 
    within_module_degree calculates: Zi = (ki-ks)/Ss 
    Ki = # de links entre un nodo i y todos los de su propio cluster Si
    Ks = promedio de links de los nodos del cluster s
    Ss = std de los links de los nodos del cluster s

    The within-module degree z-score measures how well-connected node i is to other nodes in the module.

    PAPER: Guimera, R., & Amaral, L. A. N. (2005). Functional cartography of complex metabolic networks. nature, 433(7028), 895.
    
    G: a networkx graph.
    f: a pandas dataframe.
    columna_comunidades: a column of the dataframe with the comunities for each node. If None, the comunities will be estimated using metodo comunidades.
    metodo_comunidades: method to calculate the communities in the graph G if they are not provided with columna_comunidades. 
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('Los tamaños del grafo y del dataframe no son inguales')   
    if columna_comunidades == None:
        if metodo_comunidades == "label_propagation":
            f = communities_label_propagation(G,f)
        elif metodo_comunidades == "greedy_modularity":
            f = communities_greedy_modularity(G,f)
        else:
            raise ValueError('Se necesita un metodo para hacer el clustering')
    else:
        if columna_comunidades not in f.columns:
            raise ValueError('la columna columna_comunidades no se encuentra en el dataframe')
    
    z_df = pd.DataFrame(data = {'name': [], 'within_module_degree': [] }) 
    for comunidad in set(f[columna_comunidades]):
        G2 = G.subgraph(f[f[columna_comunidades] == comunidad]["name"].values)
        Ks = 2*len(G2.edges) / len(G2.nodes)
        Ss = np.std([i[1] for i in G2.degree()])
        z_df = pd.concat([z_df,pd.DataFrame(data = {'name': list(G2.nodes), 'within_module_degree': [(i[1]-Ks)/Ss for i in G2.degree()] }) ])
    
    f = pd.merge(f, z_df, on='name')
    return f

def participation_coefficient(G,f, columna_comunidades = None, metodo_comunidades = "label_propagation"):
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
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('Los tamaños del grafo y del dataframe no son inguales')   
    if columna_comunidades == None:
        if metodo_comunidades == "label_propagation":
            f = communities_label_propagation(G,f)
        elif metodo_comunidades == "greedy_modularity":
            f = communities_greedy_modularity(G,f)
        else:
            raise ValueError('Se necesita un metodo para hacer el clustering')
    else:
        if columna_comunidades not in f.columns:
            raise ValueError('la columna columna_comunidades no se encuentra en el dataframe')
    
    p_df = pd.DataFrame(data = {'name': f['name'], 'participation_coefficient': [1 for _ in f['name']] }) 
    for nodo in f['name']:
        Kit = len(G.edges(nodo))
        for comunidad in set(f[columna_comunidades]): 
            Kis = len([edge for edge in G.edges(nodo) if edge[1] in f[ f[columna_comunidades] == comunidad ]["name"]])
            p_df.loc[ p_df["name"] == nodo, 'participation_coefficient' ] -= ( Kis / Kit ) ** 2     
    f = pd.merge(f, p_df, on='name')
    return f

def node_embeddings(G,f,dim=20, walk_length=16, num_walks=100, workers=2):
    """
    Adds the embeddings of the nodes to the dataframe f.
    G: a networkx graph.
    f: a pandas dataframe.
    dim: the dimension of the embedding.
    """
    #https://towardsdatascience.com/node2vec-embeddings-for-graph-data-32a866340fef
    #https://github.com/eliorc/Medium/blob/master/Nod2Vec-FIFA17-Example.ipynb
    #funciona con node2vec
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('Los tamaños del grafo y del dataframe no son inguales')   
    from node2vec import Node2Vec
    node2vec = Node2Vec(G, dimensions=dim, walk_length=walk_length, num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=10, min_count=1)
    
    embeddings_df = pd.DataFrame(columns = ['name']+['node_embeddings_'+str(i) for i in range(dim)])
    embeddings_df['name'] = f['name']
    for name in embeddings_df['name']:
        embeddings_df[embeddings_df['name'] == name] = [name] + list(model[str(name)])
    f = pd.merge(f, embeddings_df, on='name')
    return f





#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#--------------------- Funciones para Tweets ----------------------------------
#------------------------------------------------------------------------------

def is_RT(f, col_text):
    f["is_RT"] = pd.Series([True if "RT" in text else False for text in f[col_text]])
    return f

def topic_Detection(f, col_text, dimension, model = "LDA", stopWords = -1, custom_stopWords = [], max_df = 0.8, min_df = 0.01, lowercase = True, ngram_range = (1,3)):
    # Carga de datos
    texts = list(f[col_text])
    
    # Analisis Tf-idf y NMF
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from stopWords_castellano import diccionario_de_stopWords
    
    #las stopwords dependen del corpus
    if stopWords == -1:
        from stopWords_castellano import diccionario_de_stopWords
        stopWordsList = diccionario_de_stopWords + custom_stopWords
    elif stopWords == "en":
        from stopWords_ingles import diccionario_de_stopWords
        stopWordsList = diccionario_de_stopWords + custom_stopWords        
    
    # Genero lo vectores Nota
    count_vect = CountVectorizer(ngram_range = ngram_range, max_df = max_df, min_df = min_df, stop_words=stopWordsList,lowercase = lowercase)
    x_counts = count_vect.fit_transform(texts)
        
    # Genero matriz con valorizacion tf-idf
    tfidf_transformer = TfidfTransformer(norm = 'l2')
    x_tfidf = tfidf_transformer.fit_transform(x_counts)
    
    if model == "NMF":
        from sklearn.decomposition import NMF
        nmf = NMF(n_components = dimension)
        model_array = nmf.fit_transform(x_tfidf)
    elif model == "LDA":
        from sklearn.decomposition import LatentDirichletAllocation as LDA
        lda = LDA(n_components = dimension)
        model_array = lda.fit_transform(x_tfidf)
    elif model == "LSA":
        from sklearn.decomposition import TruncatedSVD as LSA
        lsa = LSA(n_components = dimension)
        model_array = lsa.fit_transform(x_tfidf)
    else:
        raise ValueError('model debe ser igual a "NMF", "LDA" o "LSA".')
        
    
    for dim in range(dimension):
        f['topic_'+str(dim)] = model_array[:,dim]
    return f

def emotion_Detection(f,col_text):
    from tqdm import tqdm
    import nltk
    data_emotion = pd.read_csv("dic_lexicones/data_emocion.csv")
    
    Anger = []
    Anticipation = []
    Disgust = []
    Fear = []
    Joy = []
    Sadness = []
    Surprise = []
    Trust = []
    
    for text in tqdm(f[col_text]):
        phrase  = nltk.word_tokenize(text)
        Anger_aux, Anticipation_aux, Disgust_aux, Fear_aux, Joy_aux, Sadness_aux, Surprise_aux, Trust_aux = 0,0,0,0,0,0,0,0
        for word in phrase :
            aux_df = data_emotion[data_emotion.Word_es == word]
            Anger_aux += len(aux_df.Anger) - sum(aux_df.Anger.isnull())
            Anticipation_aux += len(aux_df.Anticipation) - sum(aux_df.Anticipation.isnull())
            Disgust_aux += len(aux_df.Disgust) - sum(aux_df.Disgust.isnull())
            Fear_aux += len(aux_df.Fear) - sum(aux_df.Fear.isnull())
            Joy_aux += len(aux_df.Joy) - sum(aux_df.Joy.isnull())
            Sadness_aux += len(aux_df.Sadness) - sum(aux_df.Sadness.isnull())
            Surprise_aux += len(aux_df.Surprise) - sum(aux_df.Surprise.isnull())
            Trust_aux += len(aux_df.Trust) - sum(aux_df.Trust.isnull())
            
        Anger.append(Anger_aux)
        Anticipation.append(Anticipation_aux)
        Disgust.append(Disgust_aux)
        Fear.append(Fear_aux)
        Joy.append(Joy_aux)
        Sadness.append(Sadness_aux)
        Surprise.append(Surprise_aux)
        Trust.append(Trust_aux) 
     
    f["emot_Anger"] = Anger
    f["emot_Anticipation"] = Anticipation
    f["emot_Disgust"] = Disgust
    f["emot_Fear"] = Fear
    f["emot_Joy"] = Joy
    f["emot_Sadness"] = Sadness
    f["emot_Surprise"] = Surprise
    f["emot_Trust"] = Trust
    return f