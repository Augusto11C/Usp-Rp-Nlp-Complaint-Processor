from gensim.models import KeyedVectors
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import scipy 
import numpy as np
import logging

class yakisoba_do_chifu:
#Implementação da variação do algoritmo do restaurante chines

    def __init__(self):
        self.lista_de_palavras_tokenizadas = None
        self.model = None
        self.lista_de_vetores = []
        self.media_dos_cluster = []

    def tirar_pedido(self, lista_de_palavras_tokenizadas): 
        #Recebe lista de palavras clusterizadas
        self.lista_de_palavras_tokenizadas = lista_de_palavras_tokenizadas
    
    def prepara_os_ingredientes(self):
        #Objetivo: Carregar os embeddings na memória
        self.model = KeyedVectors.load_word2vec_format("../cbow_s300/cbow_s300.txt")
        
    def cozinhar(self):#modelo, lista de palavras
        #word_averaging
        # Objetivo: recebe uma lista de palavras e retorna uma lista de vetores
        todas_as_palavras, lista_de_vetores = set(), []
        
        for word in self.lista_de_palavras_tokenizadas:
            if isinstance(word, np.ndarray):
                self.lista_de_vetores.append(word)
            elif word in self.model.vocab:
                self.lista_de_vetores.append(self.model.syn0norm[self.model.vocab[word].index])
                todas_as_palavras.add(self.model.vocab[word].index)
            else:
                self.lista_de_vetores.append("none") #Solução para o problema de mapeamento

        if not self.lista_de_vetores:
            logging.warning("cannot compute similarity with no input %s", words)
            # FIXME: remove these examples in pre-processing
            return np.zeros(self.model.layer1_size,)

        # self.lista_de_vetores = gensim.matutils.unitvec(np.array(self.lista_de_vetores).self.lista_de_vetores(axis=0)).astype(np.float32)
        return self.lista_de_vetores

    def empratar(self, limite_de_clusters = np.inf):
        #Objetivo: Vamos executar o algoritmo do restaurante chines
        lista_de_clusters = [[]]
        n_cluster = 1
        p = 1/(1 + n_cluster)
        count = 0
        while(self.lista_de_vetores[count] == "none"):
            count = count + 1
        lista_de_clusters[0].append((self.lista_de_vetores[count],count))
        for i, vetores in enumerate(lista_de_clusters):
            if(i <= count):
                continue
            r = np.random.rand()
            if(r < p and n_cluster < lista_de_clusters):
                n_cluster = n_cluster + 1
                p = 1/(1 + n_cluster)
                lista_de_clusters.append([(vetores,i)])
            else:
                self.calcular_media_dos_clusters(lista_de_clusters) 
                for i, media_de_cluster in enumerate(self.media_dos_cluster):
                    self.media_dos_cluster[i] = cosine_similarity([media_de_cluster],[vetores])[0][0]
                #indice do argumento máximo
                lista_de_clusters[np.argmax(self.media_dos_cluster)].append((vetores,i))



    def calcular_media_dos_clusters(self, lista_de_clusters):
        self.media_dos_cluster = np.zeros(len(lista_de_clusters))
        for i, cluster in enumerate(lista_de_clusters):
                                                                        #buscamos a vetor que representa a palavra na tupla representada por vetor, existem varios vetores de cluster
            self.media_dos_cluster[i] = gensim.matutils.unitvec(np.array([vetor[0] for vetor in cluster ]).astype(np.float32))























        #return Lista de Lista de palavras clusterizadas