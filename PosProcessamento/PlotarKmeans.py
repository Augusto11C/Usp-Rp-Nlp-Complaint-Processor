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
from PalavrasMaisFrequentesPorCluster2 import PalavrasMaisFrequentesPorCluster2
import GroupedColorFunc
sys.path.insert(2, '..\TokenizadorDeCorpus')


resultados_kmeans = pickle.load(open('..\TokenizadorDeCorpus\\resultados_kmeans_word2vec_sem_java.jojo','rb'))
corpus_tokenizado = pickle.load(open('..\TokenizadorDeCorpus\corpus_tokenizado_sem_java.lai','rb'))
print(resultados_kmeans['kmeans'])
data = resultados_kmeans['data']
lista_kmeans = resultados_kmeans['kmeans']
palavrasMaisFrequentesPorCluster2 = PalavrasMaisFrequentesPorCluster2(corpus_tokenizado)
i = 5
for kmeans in lista_kmeans:
	'''if(i <= 9):
		i = i+1
		continue'''
	print(kmeans)
	lista_das_palavras_mais_frequentes_por_cLuster = palavrasMaisFrequentesPorCluster2.gerar_n_palavras_mais_frequentes_por_cluster(20,kmeans)
	Ourwordcloud().gerar_wordcloud_e_salvar(lista_palavras_mais_frequentes_clusterizadas = lista_das_palavras_mais_frequentes_por_cLuster, nome_do_arquivo = "kmeans_word2vec_sem_java_k=" + str(kmeans.n_clusters))
	#i = i+1

