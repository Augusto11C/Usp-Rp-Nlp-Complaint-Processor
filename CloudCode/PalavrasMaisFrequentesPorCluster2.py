import nltk
import pickle
import re
import os
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
import sys


sys.path.insert(0, '..\PreProcessor')
from Preprocessor import Preprocessor


class PalavrasMaisFrequentesPorCluster2:

    data = []

    
    def __init__(self,tokenized_data):
        self.data = tokenized_data


    def __get_corpus(self):
        return self.data
        

    #def get_corpus(self,)

    def __get_corpus_tokenizado_clusterizado(corpus_tokenizado, kmeans):
        print("clusterizando corpus")
        corpus_clusterizado = [[] for _ in range(kmeans.n_clusters)]
        for index, n_cluster in enumerate(kmeans.labels_):
            corpus_clusterizado[n_cluster] = corpus_clusterizado[n_cluster] + corpus_tokenizado[index]
            if(index%10000==0):
                print(index)
        return corpus_clusterizado

        
        #recebe uma lista simples de palavras e devolve uma lista de tuplas(palavra, frequencia)
    def __get_tupla_frequencia_palavras(lista_palavras): 
        print("ordenando palavras")
        lista = sorted(lista_palavras)
        print("gerando tuplas")
        lista_tuplas = []
        contador = 1
        atual = lista[0]
        for palavra in lista:
            if(atual == palavra):
                contador+=1
                continue
            lista_tuplas.append((atual, contador))
            atual = palavra
            contador = 1
        lista_tuplas.append((atual, contador))
        print(lista_tuplas)
        return sorted(lista_tuplas, key = lambda tupla: tupla[1], reverse = True)

      #recebe um numero n e lista de listas de palavras e retorna lista de listas das n palavras mais frequentes de cada cluster
    def __get_n_palavras_mais_frequentes_cluster(n, lista_palavras_clusterizadas):
        print("gerando n palavras mais frequentes")
        lista_mais_frequentes = [[] for _ in range(len(lista_palavras_clusterizadas))]
        for n_cluster,cluster in enumerate(lista_palavras_clusterizadas):
            tuplas = PalavrasMaisFrequentesPorCluster2.__get_tupla_frequencia_palavras(cluster)
            #print(tuplas)
            cont = 0
            for contador,tupla in enumerate(tuplas):

                if(contador == n):
                    break
                lista_mais_frequentes[n_cluster].append(tupla)
        
        return lista_mais_frequentes

        #método principal que recebe numero n e caminho path e devolve lista de listas das palavras ,ais frequentes por cluster
    def gerar_n_palavras_mais_frequentes_por_cluster(self,n, kmeans):
        corpus_tokenizado = PalavrasMaisFrequentesPorCluster2.__get_corpus(self)
        palavras_clusterizadas = PalavrasMaisFrequentesPorCluster2.__get_corpus_tokenizado_clusterizado(corpus_tokenizado, kmeans)
        return PalavrasMaisFrequentesPorCluster2.__get_n_palavras_mais_frequentes_cluster(n, palavras_clusterizadas)



###########################################################################################################################################################################
###########################################################################################################################################################################



    def get_corpus_tokenizado_clusterizado(self, corpus_tokenizado, kmeans):
        print("clusterizando corpus")
        corpus_clusterizado = [[] for _ in range(kmeans.n_clusters)]
        for index, n_cluster in zip([i for i in range(len(kmeans.labels_))] ,kmeans.labels_):
            corpus_clusterizado[n_cluster].append(corpus[index])
        return corpus_clusterizado

        
        #recebe uma lista simples de palavras e devolve uma lista de tuplas(palavra, frequencia)
    def get_tupla_frequencia_palavras(self, lista_palavras): 
        print("ordenando palavras")
        lista = sorted(lista_palavras)
        print("gerando tuplas")
        lista_tuplas = []
        contador = 1
        atual = lista[0]
        for palavra in lista:
            if(atual == palavra):
                contador+=1
                continue
            lista_tuplas.append((atual, contador))
            atual = palavra
            contador = 1
        lista_tuplas.append((atual, contador))

        return sorted(lista_tuplas, key = lambda tupla: tupla[1], reverse = True)

      #recebe um numero n e lista de listas de palavras e retorna lista de listas das n palavras mais frequentes de cada cluster
    def get_n_palavras_mais_frequentes_cluster(self, n, lista_palavras_clusterizadas):
        print("gerando n palavras mais frequentes")
        lista_mais_frequentes = [[] for _ in range(len(lista_palavras_clusterizadas))]
        for n_cluster,cluster in enumerate(lista_palavras_clusterizadas):
            tuplas = PalavrasMaisFrequentesPorCluster2.__get_tupla_frequencia_palavras(cluster)
            print(tuplas)
            cont = 0
            for contador,tupla in enumerate(tuplas):

                if(contador == n):
                    break
                lista_mais_frequentes[n_cluster].append(tupla)
        
        return lista_mais_frequentes