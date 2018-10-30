import matplotlib.pyplot as plt
import os
import random
import GroupedColorFunc
from wordcloud import WordCloud, get_single_color_func
import PalavrasMaisFrequentesPorCluster

class Ourwordcloud():

    
    def get_lista_cores(self):
        cores = ['midnightblue','salmon','red','green','purple','yellow','cyan', 'pink', 'brown']
        return cores

    def colorir_palavras_de_um_cluster(self, lista_palavras_mais_frequentes_do_cluster, cor):
        color_to_words = {} 
        color_to_words[cor] = lista_palavras_mais_frequentes_do_cluster
        return color_to_words

    # def multiplicacao_das_palavras(self, lista_palavras_mais_frequentes_clusterizadas):
    #     text = ""
    #     for  palavra, frequencia in lista_palavras_mais_frequentes_clusterizadas:
    #         print(palavra,frequencia)
    #         if frequencia > 2 :
    #             for i in range(frequencia):
    #                 text = text + " " + palavra
    #                 print(text)
    #     return text 
    # 
    #  
    def multiplicacao_das_palavras(self, lista_palavras_mais_frequentes_clusterizadas):
        text = ""
        for  palavra, frequencia in lista_palavras_mais_frequentes_clusterizadas:
            text = text + " " + palavra
        return text


    def gerar_wordcloud_e_salvar(self, lista_palavras_mais_frequentes_clusterizadas, nome_do_arquivo):# lista_palavras_mais_frequentes_clusterizados [[(palavra1_cluster1,freq)],[(palavra1_cluster2,freq)]]
        default_color = 'grey'
        colors = self.get_lista_cores()
        for cluster, palavras_do_cluster in enumerate(lista_palavras_mais_frequentes_clusterizadas):
            cor = random.choice(colors)
            colors.remove(cor)
            #print("Antes Concantenar Palavras")
            palavras_concatenadas = self.multiplicacao_das_palavras(palavras_do_cluster)
            #print(palavras_concatenadas)
            #teste = [palav[0] for palav in palavras_do_cluster]
            #print(teste)
            #color_to_words = self.colorir_palavras_de_um_cluster(teste, cor)
            color_to_words = self.colorir_palavras_de_um_cluster(palavras_concatenadas, cor)
            #print(color_to_words)
            grouped_color_func = GroupedColorFunc.GroupedColorFunc(color_to_words, cor)
            #print(grouped_color_func)
            wc = WordCloud(width = 2000,height = 1800, background_color = 'white', min_font_size = 5, relative_scaling = 0.5,  ).generate(palavras_concatenadas)
            #print("Depois WordCloud")
            
            wc.recolor(color_func = grouped_color_func)
            plt.figure()
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            # plt.show()
            wc.to_file(nome_do_arquivo + "_cluster_" + str(cluster+1) + ".png")
        
