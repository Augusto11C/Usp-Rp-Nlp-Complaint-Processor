import nltk
import pickle
import re
import os
from sklearn.cluster import KMeans
import sys

sys.path.insert(0, '..\PreProcessor')
from Preprocessor import Preprocessor


class PalavrasMaisFrequentesPorCluster:

    data = []
    
    def __init__(self,data):
        self.data = data


    def __get_corpus(self):
        if not self.data:
            stopwordz = nltk.corpus.stopwords.words('portuguese')

            for d in os.listdir("../../CidadaoData/2017/Dezembro"):
                dict = pickle.load(open("../../CidadaoData/2017/Dezembro/"+d,"rb"), encoding="utf-8")
                self.data.append(dict["reclamacao"].strip().lower())
        return self.data
        

    #recebe uma lista do corpus e um objeto kmeans e devolve uma lista de listas de textos, em que cada lista interna representa um cluster
    def __get_corpus_clusterizado(corpus, kmeans):
        corpus_clusterizado = [[] for _ in range(kmeans.n_clusters)]
        for index, n_cluster in zip([i for i in range(len(kmeans.labels_))] ,kmeans.labels_):
            corpus_clusterizado[n_cluster].append(corpus[index])
        return corpus_clusterizado

        """recebe a lista de listas de textos, transforma a lista de textos em lista de palavras 
    e devolve lista de listas de palavras, em que cada lista interna representa um cluster"""
    def __get_palavras_clusterizadas(corpus_clusterizado):
        '''palavras_clusterizadas = [[] for _ in range(len(corpus_clusterizado))]
        for n_cluster,cluster in enumerate(corpus_clusterizado):
            string = ""
            for texto in cluster:
                string = string + " " + texto
                palavras = string.split(" ")
                for palavra in palavras:
                    palavras_clusterizadas[n_cluster].append(palavra)
        #return palavras_clusterizadas
        stopwordz = nltk.corpus.stopwords.words('portuguese')+["","\r\n","pois","que","pra","ter","fazer","ser","para"]        
        
        palavras_clusterizadas = [[] for _ in range(len(corpus_clusterizado))]
        for n_cluster,cluster in enumerate(corpus_clusterizado):
            string = ""
            for texto in cluster:
                string = string + " " + texto
                palavras = re.sub('[^ a-zA-Z0-9]', ' ', string)
        
            palavras = palavras.split(" ")
            
            for palavra in palavras:
                word = palavra.replace("\\s+","")
                
                if(word not in stopwordz and not len(word) <= 2 and not word.isdigit()):
                    palavras_clusterizadas[n_cluster].append(word)'''
        preprocessor = Preprocessor
        palavras_clusterizadas = [[] for _ in range(len(corpus_clusterizado))]
        for n_cluster,cluster in enumerate(corpus_clusterizado):
            palavras_tokenizadas = []
            for texto in cluster:
                texto_preprocessado = preprocessor.preprocess(texto)
                palavras_tokenizadas = palavras_tokenizadas + preprocessor.tokenize_string(texto_preprocessado)
            palavras_clusterizadas[n_cluster] = palavras_clusterizadas[n_cluster] + palavras_tokenizadas

        return palavras_clusterizadas

        #recebe uma lista simples de palavras e devolve uma lista de tuplas(palavra, frequencia)
    def __get_tupla_frequencia_palavras(lista_palavras): 
        lista = sorted(lista_palavras)
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
    def __get_n_palavras_mais_frequentes_cluster(n, lista_palavras_clusterizadas):
        lista_mais_frequentes = [[] for _ in range(len(lista_palavras_clusterizadas))]
        for n_cluster,cluster in enumerate(lista_palavras_clusterizadas):
            tuplas = PalavrasMaisFrequentesPorCluster.__get_tupla_frequencia_palavras(cluster)
            print(tuplas)
            cont = 0
            for contador,tupla in enumerate(tuplas):

                if(contador == n):
                    break
                lista_mais_frequentes[n_cluster].append(tupla)
        
        return lista_mais_frequentes

        #método principal que recebe numero n e caminho path e devolve lista de listas das palavras ,ais frequentes por cluster
    def gerar_n_palavras_mais_frequentes_por_cluster(self,n, kmeans):
        corpus = PalavrasMaisFrequentesPorCluster.__get_corpus(self)
        corpus_clusterizado = PalavrasMaisFrequentesPorCluster.__get_corpus_clusterizado(corpus,kmeans)
        palavras_clusterizadas = PalavrasMaisFrequentesPorCluster.__get_palavras_clusterizadas(corpus_clusterizado)
        return PalavrasMaisFrequentesPorCluster.__get_n_palavras_mais_frequentes_cluster(n, palavras_clusterizadas)

    def get_corpus_clusterizado(self, corpus, kmeans):
        corpus_clusterizado = [[] for _ in range(kmeans.n_clusters)]
        for index, n_cluster in zip([i for i in range(len(kmeans.labels_))] ,kmeans.labels_):
            corpus_clusterizado[n_cluster].append(corpus[index])
        return corpus_clusterizado

        """recebe a lista de listas de textos, transforma a lista de textos em lista de palavras 
    e devolve lista de listas de palavras, em que cada lista interna representa um cluster"""
    def get_palavras_clusterizadas(self, corpus_clusterizado):
        '''palavras_clusterizadas = [[] for _ in range(len(corpus_clusterizado))]
        for n_cluster,cluster in enumerate(corpus_clusterizado):
            string = ""
            for texto in cluster:
                string = string + " " + texto
                palavras = string.split(" ")
                for palavra in palavras:
                    palavras_clusterizadas[n_cluster].append(palavra)
        #return palavras_clusterizadas
        stopwordz = nltk.corpus.stopwords.words('portuguese')+["","\r\n","pois","que","pra","ter","fazer","ser","para"]        
        
        palavras_clusterizadas = [[] for _ in range(len(corpus_clusterizado))]
        for n_cluster,cluster in enumerate(corpus_clusterizado):
            string = ""
            for texto in cluster:
                string = string + " " + texto
                palavras = re.sub('[^ a-zA-Z0-9]', ' ', string)
        
            palavras = palavras.split(" ")
            
            for palavra in palavras:
                word = palavra.replace("\\s+","")
                
                if(word not in stopwordz and not len(word) <= 2 and not word.isdigit()):
                    palavras_clusterizadas[n_cluster].append(word)'''
        preprocessor = Preprocessor
        palavras_clusterizadas = [[] for _ in range(len(corpus_clusterizado))]
        for n_cluster,cluster in enumerate(corpus_clusterizado):
            palavras_tokenizadas = []
            for texto in cluster:
                texto_preprocessado = preprocessor.preprocess(texto)
                palavras_tokenizadas = palavras_tokenizadas + preprocessor.tokenize_string(texto_preprocessado)
            palavras_clusterizadas[n_cluster] = palavras_clusterizadas[n_cluster] + palavras_tokenizadas

        return palavras_clusterizadas

    def get_tupla_frequencia_palavras(self,lista_palavras): 
        lista = sorted(lista_palavras)
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
        lista_mais_frequentes = [[] for _ in range(len(lista_palavras_clusterizadas))]
        for n_cluster,cluster in enumerate(lista_palavras_clusterizadas):
            tuplas = PalavrasMaisFrequentesPorCluster.get_tupla_frequencia_palavras(self, cluster)
            print(tuplas)
            cont = 0
            for contador,tupla in enumerate(tuplas):

                if(contador == n):
                    break
                lista_mais_frequentes[n_cluster].append(tupla)
        
        return lista_mais_frequentes

