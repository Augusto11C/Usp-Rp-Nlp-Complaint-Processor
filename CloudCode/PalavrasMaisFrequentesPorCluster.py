import nltk
import pickle
import re
import os
from sklearn.cluster import KMeans


class PalavrasMaisFrequentesPorCluster:
 
    def __get_corpus():
        data = []

        stopwordz = nltk.corpus.stopwords.words('portuguese')

        for d in os.listdir("../../CidadaoData/2017/Dezembro"):
            dict = pickle.load(open("../../CidadaoData/2017/Dezembro/"+d,"rb"), encoding="utf-8")
            data.append(dict["reclamacao"].strip().lower())
        return data

    #recebe uma lista do corpus e um objeto kmeans e devolve uma lista de listas de textos, em que cada lista interna representa um cluster
    def __get_corpus_clusterizado(corpus, kmeans):
        corpus_clusterizado = [[] for _ in range(kmeans.n_clusters)]
        for index, n_cluster in zip([i for i in range(len(kmeans.labels_))] ,kmeans.labels_):
            corpus_clusterizado[n_cluster].append(corpus[index])
        return corpus_clusterizado

        """recebe a lista de listas de textos, transforma a lista de textos em lista de palavras 
    e devolve lista de listas de palavras, em que cada lista interna representa um cluster"""
    def __get_palavras_clusterizadas(corpus_clusterizado):
        palavras_clusterizadas = [[] for _ in range(len(corpus_clusterizado))]
        for n_cluster,cluster in enumerate(corpus_clusterizado):
            string = ""
            for texto in cluster:
                string = string + " " + texto
                palavras = string.split(" ")
                for palavra in palavras:
                    palavras_clusterizadas[n_cluster].append(palavra)
        #return palavras_clusterizadas
        stopwordz = nltk.corpus.stopwords.words('portuguese')+["","\r\n","pois","que","pra","ter","fazer","ser","para"]        
        #stopWords = [x for x in ENGLISH_STOP_WORDS]
        '''otherCommonWords = ['make','year','years','new','people','said','say','time','brown','good','told','000','says','took','way','think','going','just','don','did','use','best','didn', 'mln', 'cts', 'net', 'dlrs', 'shr', 'blah', 'revs', 'qtr', 'oper', 'march', 'bank', 'company', 'corp', 'sales', 'dlr', 'billion', 'stg', 'loss', 'profit', 'revs', 'div', 'pct', 'record', 'prior', 'pay', 'qtly', 'dividend', '4th', 'note', 'sets', 'avg', 'shrs', 'includes', 'quarterly', 'share', 'shares', 'miles', 'mths', 'april', 'february', 'stock', 'prices', 'price', 'market', 'government', 'exchange', 'january', 'york', 'week', 'quarter', 'december', 'added', 'production', 'bbl', 'feb', 'official', 'international', 'deficit', 'raises', 'debit', 'trade', 'baker', 'rate', 'crude', 'tax', 'debt', 'debts', 'money', 'business', 'offer', 'foreign', 'contract', 'agreement', 'systems', 'board', '1st', '2nd', '3rd', 'commercial', 'dollar', 'dollars', 'excludes', 'extraordinary', 'securities', 'trading', 'economic', 'current', 'financial', 'issue', 'today', 'rose', 'expected', 'dec', 'jan', 'gain', 'declared', 'months', 'payable', 'available', 'income', 'operations', 'regular', 'traders', 'revenue', 'national', 'world', 'effective', 'wti', 'making', 'sale', 'results', 'periods', 'respectively', 'gain', 'month', 'common', 'credit', 'buy', 'public', 'initial', 'talks', 'total', 'bond', 'expects', 'sell', 'twa', 'averager', 'ended', 'forth', 'compared', 'period', 'sees', 'ago', 'fiscal', 'budget', 'end', 'department', 'day', 'group', 'cash', 'earnings', 'include', 'exclude', 'june', 'pre', 'rev', 'fall', 'raise', 'agreed', 'fourth', 'proceeds', 'american', 'output', 'president', 'qtlys', 'analysts', 'tonnes']
        for w in otherCommonWords:
            stopWords.append(w)'''
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
                    palavras_clusterizadas[n_cluster].append(word)
            
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
    def gerar_n_palavras_mais_frequentes_por_cluster(n, kmeans):
        corpus = PalavrasMaisFrequentesPorCluster.__get_corpus()
        corpus_clusterizado = PalavrasMaisFrequentesPorCluster.__get_corpus_clusterizado(corpus,kmeans)
        palavras_clusterizadas = PalavrasMaisFrequentesPorCluster.__get_palavras_clusterizadas(corpus_clusterizado)
        return PalavrasMaisFrequentesPorCluster.__get_n_palavras_mais_frequentes_cluster(n, palavras_clusterizadas)