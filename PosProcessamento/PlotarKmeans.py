import nltk
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from sklearn.metrics import silhouette_score
import sys
import re

sys.path.insert(0, '..\CloudCode')
from Ourwordcloud import Ourwordcloud
sys.path.insert(1, '..\Preprocessor')
from Preprocessor import Preprocessor
# import Ourwordcloud
from PalavrasMaisFrequentesPorCluster import PalavrasMaisFrequentesPorCluster
import GroupedColorFunc

resultados_kmeans = pickle.load(open('resultados_kmeans.jojo','rb'))
print(resultados_kmeans['kmeans'])
data = resultados_kmeans['data']
lista_kmeans = resultados_kmeans['kmeans']
palavrasMaisFrequentesPorCluster = PalavrasMaisFrequentesPorCluster(data)
for kmeans in lista_kmeans:
	print(kmeans)
	tuplas_mais_frequentes = palavrasMaisFrequentesPorCluster.gerar_n_palavras_mais_frequentes_por_cluster(20,kmeans)
	Ourwordcloud().gerar_wordcloud_e_salvar(lista_palavras_mais_frequentes_clusterizadas = lista_das_palavras_mais_frequentes_por_cLuster, nome_do_arquivo = "kmeans_tfidf_k=" + kmeans.n_cluster)


