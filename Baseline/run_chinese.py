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

sys.path.insert(0,"..\PreProcessor")
from Preprocessor import Preprocessor


sys.path.insert(0, '..\RestauranteChinesDePalavras')
from yakisoba_do_chifu import yakisoba_do_chifu

#mega texto tokeniza

#dict_keys(['data', 'kmeans', 'scores'])



kmeans_normal = pickle.load(open("../../KMEANS_BASE/resultados_kmeans.jojo","rb"))

#kmeans_w2c = pickle.load(open("../../KMEANS_BASE/resultados_kmeans_word2vec.jojo"))
kmeans_do_arqKmeans = kmeans_normal['kmeans']
data_do_arqKmeans = kmeans_normal['data']

# obj_clusterizado = "augusto"
# obj_clusterizado = PalavrasMaisFrequentesPorCluster(data_do_arqKmeans).get_corpus_clusterizado(data_do_arqKmeans, kmeans_do_arqKmeans[1])


# palavras_clusterizadas = PalavrasMaisFrequentesPorCluster(data_do_arqKmeans).get_palavras_clusterizadas(obj_clusterizado)
# print(palavras_clusterizadas[1])

# for kmean_aq in kmeans_do_arqKmeans:
#     palavras_clusterizadas = PalavrasMaisFrequentesPorCluster(data_do_arqKmeans).get_palavras_clusterizadas(data_do_arqKmeans,obj_clusterizado)
#     for lista_palavras in palavras_clusterizadas:
#         yakisoba_do_chifu = yakisoba_do_chifu()
#         yakisoba_do_chifu.tirar_pedido(self, lista_de_palavras_tokenizadas):
#         yakisoba_do_chifu.prepara_os_ingredientes()
#         yakisoba_do_chifu.cozinhar()
#         yakisoba_do_chifu.empratar()
#         lista_de_cluster = yakisoba_do_chifu.calcular_media_dos_clusters()
#         PalavrasMaisFrequentesPorCluster(data_aq).get_n_palavras_mais_frequentes_cluster(20,lista_de_cluster)


num_kmeans = int(sys.argv[1])

obj_clusterizado = PalavrasMaisFrequentesPorCluster(data_do_arqKmeans).get_corpus_clusterizado(data_do_arqKmeans, kmeans_do_arqKmeans[num_kmeans])
palavras_clusterizadas = PalavrasMaisFrequentesPorCluster(data_do_arqKmeans).get_palavras_clusterizadas(obj_clusterizado)
for lista_palavras in palavras_clusterizadas:
        yakisoba_do_chifu = yakisoba_do_chifu()
        yakisoba_do_chifu.tirar_pedido(lista_palavras)
        yakisoba_do_chifu.prepara_os_ingredientes()
        yakisoba_do_chifu.cozinhar()
        yakisoba_do_chifu.empratar()
        lista_de_cluster = yakisoba_do_chifu.calcular_media_dos_clusters()
        PalavrasMaisFrequentesPorCluster(data_aq).get_n_palavras_mais_frequentes_cluster(20,lista_de_cluster)
