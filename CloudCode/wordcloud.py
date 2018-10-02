import matplotlib.pyplot as plt
import os
import random
import GroupedColorFunc
from wordcloud import WordCloud, get_single_color_func

class SimpleGroupedColorFunc(object):

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)

    def get_lista_cores(self):
        cores = ['midnightblue','salmon','red','green','purple','yellow','cyan']
        return cores

    def colorir_palavras_de_um_cluster(self, lista_palavras_mais_frequentes_do_cluster, cor):
        color_to_words = {} 
        color_to_words[cor] = lista_palavras_mais_frequentes_do_cluster
        return color_to_words

    def multiplicacao_das_palavras(self, lista_palavras_mais_frequentes_clusterizadas):
        text = ""
        for  palavra, frequencia in lista_palavras_mais_frequentes_clusterizadas:
            for i in range(frequencia):
                text = text + " " + palavra
        return text  

    def gerar_wordcloud_e_salvar(self, lista_palavras_mais_frequentes_clusterizadas, nome_do_arquivo):# lista_palavras_mais_frequentes_clusterizados [[(palavra1_cluster1,freq)],[(palavra1_cluster2,freq)]]
        default_color = 'grey'
        colors = self.get_lista_cores()
        for cluster, palavras_do_cluster in enumerate(lista_palavras_mais_frequentes_clusterizadas):
            cor = random.choice(colors)
            colors.remove(cor)
            palavras_concatenadas = self.multiplicacao_das_palavras(palavras_do_cluster)
            color_to_words = self.colorir_palavras_de_um_cluster(palavras_concatenadas, cor)
            grouped_color_func = GroupedColorFunc.GroupedColorFunc(palavras_concatenadas, default_color)

            wc = WordCloud(width = 2000,height = 1800, background_color = 'white', min_font_size = 5, relative_scaling = 1 ).generate(palavras_concatenadas)
            wc.recolor(color_func = grouped_color_func)
            plt.figure()
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            wc.to_file(nome_do_arquivo + "_cluster_" + str(cluster+1) + ".png")
        
