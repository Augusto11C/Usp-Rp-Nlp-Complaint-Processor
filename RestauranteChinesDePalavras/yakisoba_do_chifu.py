from gensim.models import KeyedVectors
import gensim
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import scipy 
import numpy as np
import logging
import sys

class yakisoba_do_chifu:
#Implementação da variação do algoritmo do restaurante chines

    def __init__(self):
        self.lista_de_palavras_tokenizadas = None
        self.model = None
        self.lista_de_vetores = []
        self.media_dos_cluster = []
        self.lista_de_vetores =[]

    def tirar_pedido(self, lista_de_palavras_tokenizadas): 
        #Recebe lista de palavras clusterizadas
        self.lista_de_palavras_tokenizadas = lista_de_palavras_tokenizadas
    
    def prepara_os_ingredientes(self):
        #Objetivo: Carregar os embeddings na memória
        self.model = KeyedVectors.load_word2vec_format("../skip_s300/skip_s300.txt")
        self.model.init_sims(replace=True)
        
    def cozinhar(self):#modelo, lista de palavras
        #word_averaging
        # Objetivo: recebe uma lista de palavras e retorna uma lista de vetores
        todas_as_palavras = set()
            
        for word in self.lista_de_palavras_tokenizadas:
            if isinstance(word, np.ndarray):
                self.lista_de_vetores.append(word)
            elif word in self.model.vocab:
                self.lista_de_vetores.append(self.model.syn0norm[self.model.vocab[word].index])
                todas_as_palavras.add(self.model.vocab[word].index)
            else:
                placeholder = np.zeros(300)
                placeholder.fill(np.nan)
                self.lista_de_vetores.append(placeholder) #Solução para o problema de mapeamento

        if not self.lista_de_vetores:
            logging.warning("cannot compute similarity with no input %s", '2')
            # FIXME: remove these examples in pre-processing
            # return np.zeros(self.model.layer1_size)

        # self.lista_de_vetores = gensim.matutils.unitvec(np.array(self.lista_de_vetores).self.lista_de_vetores(axis=0)).astype(np.float32)
        return self.lista_de_vetores

    def empratar(self, limite_de_clusters = np.inf):
        #Objetivo: Vamos executar o algoritmo do restaurante chines
        self.lista_de_clusters = [[]]
        n_cluster = 1
        p = 1/(1 + n_cluster)
        count = 0
        placeholder = np.zeros(300)
        placeholder.fill(np.nan)
        try:
            while((self.lista_de_vetores[count] == placeholder).all()):
                count = count + 1
        except:
            print("DEU ERRO")
            print(self.lista_de_vetores)
            print("*************")
            print("Count: " + count + " =-=-=-=-=-=-=-=-=-=-=-=-")
            print(placeholder)
            sys.exit(-1)
        self.lista_de_clusters[0].append((self.lista_de_vetores[count],count))
        for i, vetores in enumerate(self.lista_de_vetores):

            if(i <= count):
                continue

            if(self.is_there_nan(vetores) is True):
                print("detector de nan 2")
                continue


            r = np.random.rand()
            if((r < p) and n_cluster < limite_de_clusters):
                print("Criou novo cluster")
                n_cluster = n_cluster + 1
                p = 1/(1 + n_cluster)
                self.lista_de_clusters.append([(vetores,i)])
            else:
                print("Vai inserir em um cluster existente")
                self.calcular_media_dos_clusters() 
                lista_aux_media = [0 for _ in range(len(self.media_dos_cluster))]
                for ii, media_de_cluster in enumerate(self.media_dos_cluster):
                    uuuu=cosine_similarity([media_de_cluster],[vetores])[0][0]
                    #print(uuuu)
                    lista_aux_media[ii] = uuuu

                #indice do argumento máximo
                #print("Arg max>" + str(np.argmax(lista_aux_media)))
                print("Len>" + str(len(self.lista_de_clusters)))
                #print(self.media_dos_cluster)
                self.lista_de_clusters[np.argmax(lista_aux_media)].append((vetores,i))

        return self.lista_de_clusters

    def is_there_nan(self, vetor):
#        def is_there_nan(self, vetor):
        for element in vetor:
            if(np.isnan(element)):
                return True
        return False



    def faz_sobremesa(self):
        lista_de_palavras = []
        for clusterzz in self.lista_de_clusters:
            indices = [x[1] for x in clusterzz]
            palavras = [self.lista_de_palavras_tokenizadas[j] for j in indices]
            lista_de_palavras.append(palavras)

        return lista_de_palavras

    def calcular_media_dos_clusters(self):
        #KEEP None
        self.media_dos_cluster = [None for _ in range(len(self.lista_de_clusters))]
        for i, cluster in enumerate(self.lista_de_clusters):

            my_np = np.array([vetor[0] for vetor in cluster])
            #print("my_np")
            #print(my_np)    

            my_np_media = my_np.mean(axis=0).astype(np.float32)
            #print("my_np_media")

            #print(my_np_media)
                                                                        #buscamos a vetor que representa a palavra na tupla representada por vetor, existem varios vetores de cluster
            self.media_dos_cluster[i] = gensim.matutils.unitvec(np.array([vetor[0] for vetor in cluster]).mean(axis=0).astype(np.float32))

            #print('self.media_dos_cluster[i]')
            #print(self.media_dos_cluster[i])

        self.media_dos_cluster= np.array(self.media_dos_cluster)




















        #return Lista de Lista de palavras clusterizadas