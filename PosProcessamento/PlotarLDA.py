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
sys.path.insert(2, '..\LDA')

n = 5

for i in range(6,11):
	list_LDA = pickle.load(open("..\LDA\TOP_WORDS_DECREASING_LDA_REG_1000_" + str(i) + ".list", "rb"))
	tuple = [[]for _ in range(i)]
	for j,list in enumerate(list_LDA):
		l = 20
		for w in list:
			k = j*l*20
			tuple[j].append((w,k))
	Ourwordcloud().gerar_wordcloud_e_salvar(lista_palavras_mais_frequentes_clusterizadas = tuple, nome_do_arquivo = "LDA_" + str(i)) 


		
	
	'''j = 800
	for word in list:
		tuple[i-n-1].append((word,j))
		j = j - 40
	i = i + 1
Ourwordcloud().gerar_wordcloud_e_salvar(lista_palavras_mais_frequentes_clusterizadas = tuple, nome_do_arquivo = "LDA")'''



