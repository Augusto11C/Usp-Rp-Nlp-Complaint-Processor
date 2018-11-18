import nltk
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from sklearn.metrics import silhouette_score
import sys

sys.path.insert(0, '..\CloudCode')
from Ourwordcloud import Ourwordcloud
# import Ourwordcloud
from PalavrasMaisFrequentesPorCluster import PalavrasMaisFrequentesPorCluster
import GroupedColorFunc

sys.path.insert(0, '..\RestauranteChinesDePalavras')
from yakisoba_do_chifu import yakisoba_do_chifu

data = []
stopwordz = nltk.corpus.stopwords.words('portuguese')

for d in os.listdir("../../CidadaoData/2017/Dezembro"):
	dict = pickle.load(open("../../CidadaoData/2017/Dezembro/"+d,"rb"), encoding="utf-8")
	data.append(dict["reclamacao"].strip().lower())

CV = CountVectorizer(analyzer="word",preprocessor=None,stop_words=stopwordz) 

td = CV.fit_transform(text for text in data)

kmeans = KMeans(n_clusters=9, verbose=1).fit(td)

TV = TfidfVectorizer(analyzer="word",preprocessor=None,stop_words=stopwordz)

td = TV.fit_transform(text for text in data)

kmeans = KMeans(n_clusters=9, verbose=1).fit(td)

labels = kmeans.labels_
print(metrics.silhouette_score(td, labels, metric='euclidean'))

obj_clusterizado = PalavrasMaisFrequentesPorCluster(data).get_corpus_clusterizado(data,kmeans)
palavras_clusterizadas = PalavrasMaisFrequentesPorCluster(data).get_palavras_clusterizadas(obj_clusterizado)

lista_de_palavras_tokenizadas = palavras_clusterizadas[0]

yakisoba_do_chifu = yakisoba_do_chifu()
yakisoba_do_chifu.tirar_pedido(lista_de_palavras_tokenizadas)
yakisoba_do_chifu.prepara_os_ingredientes()
yakisoba_do_chifu.cozinhar()
yakisoba_do_chifu.empratar()
yakisoba_do_chifu.calcular_media_dos_clusters()

#listaKappa = [[("augusto",20)], [("lucas",10)], [("Fernado",5)]]
#Ourwordcloud().gerar_wordcloud_e_salvar(lista_palavras_mais_frequentes_clusterizadas = lista_das_palavras_mais_frequentes_por_cLuster, nome_do_arquivo = "wordCloud")

