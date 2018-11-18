import pickle
import sys
import os
import numpy as np
from sklearn.cluster import KMeans

#rodar com um argumento de 5 até 11 (nro de clusters)
#rodar no diretorio com as reclamacoes de 2017

jojo = pickle.load(open('resultados_kmeans.jojo','rb'))
kmeans = jojo['kmeans'][int(sys.argv[1]) - 5]
n_clusters = kmeans.n_clusters
labels = kmeans.labels_

cluster_0 = {
	'atendimento' : 0,
	'telecomunicacoes' : 0,
	'saude' : 0,
	'educacao_publica' : 0,
	'infraestrutura' : 0,
	'meio_ambiente' : 0,
	'transporte' : 0,
	'governamental' : 0,
	'total' : 0
}

cluster_1 = {
	'atendimento' : 0,
	'telecomunicacoes' : 0,
	'saude' : 0,
	'educacao_publica' : 0,
	'infraestrutura' : 0,
	'meio_ambiente' : 0,
	'transporte' : 0,
	'governamental' : 0,
	'total' : 0
}

cluster_2 = {
	'atendimento' : 0,
	'telecomunicacoes' : 0,
	'saude' : 0,
	'educacao_publica' : 0,
	'infraestrutura' : 0,
	'meio_ambiente' : 0,
	'transporte' : 0,
	'governamental' : 0,
	'total' : 0
}

cluster_3 = {
	'atendimento' : 0,
	'telecomunicacoes' : 0,
	'saude' : 0,
	'educacao_publica' : 0,
	'infraestrutura' : 0,
	'meio_ambiente' : 0,
	'transporte' : 0,
	'governamental' : 0,
	'total' : 0
}

cluster_4 = {
	'atendimento' : 0,
	'telecomunicacoes' : 0,
	'saude' : 0,
	'educacao_publica' : 0,
	'infraestrutura' : 0,
	'meio_ambiente' : 0,
	'transporte' : 0,
	'governamental' : 0,
	'total' : 0
}

cluster_5 = {
	'atendimento' : 0,
	'telecomunicacoes' : 0,
	'saude' : 0,
	'educacao_publica' : 0,
	'infraestrutura' : 0,
	'meio_ambiente' : 0,
	'transporte' : 0,
	'governamental' : 0,
	'total' : 0
}

cluster_6 = {
	'atendimento' : 0,
	'telecomunicacoes' : 0,
	'saude' : 0,
	'educacao_publica' : 0,
	'infraestrutura' : 0,
	'meio_ambiente' : 0,
	'transporte' : 0,
	'governamental' : 0,
	'total' : 0
}

cluster_7 = {
	'atendimento' : 0,
	'telecomunicacoes' : 0,
	'saude' : 0,
	'educacao_publica' : 0,
	'infraestrutura' : 0,
	'meio_ambiente' : 0,
	'transporte' : 0,
	'governamental' : 0,
	'total' : 0
}

cluster_8 = {
	'atendimento' : 0,
	'telecomunicacoes' : 0,
	'saude' : 0,
	'educacao_publica' : 0,
	'infraestrutura' : 0,
	'meio_ambiente' : 0,
	'transporte' : 0,
	'governamental' : 0,
	'total' : 0
}

cluster_9 = {
	'atendimento' : 0,
	'telecomunicacoes' : 0,
	'saude' : 0,
	'educacao_publica' : 0,
	'infraestrutura' : 0,
	'meio_ambiente' : 0,
	'transporte' : 0,
	'governamental' : 0,
	'total' : 0
}

cluster_10 = {
	'atendimento' : 0,
	'telecomunicacoes' : 0,
	'saude' : 0,
	'educacao_publica' : 0,
	'infraestrutura' : 0,
	'meio_ambiente' : 0,
	'transporte' : 0,
	'governamental' : 0,
	'total' : 0
}


lista = [cluster_0,cluster_1,cluster_2,cluster_3,cluster_4,cluster_5,cluster_6,cluster_7,cluster_8,cluster_9,cluster_10]

cont = 0

for month in os.listdir("./"):
	if(os.path.isdir("./"+month)):
		for waifu in os.listdir("./"+month):
			with open("./"+month+"/"+waifu, mode='rb') as kawaii:
				arq = pickle.load(kawaii)
				arq_categoria = arq['categoria']
				indice = labels[cont]
				cluster = lista[indice]
				cluster[arq_categoria] = cluster[arq_categoria] + 1
				cluster['total'] = cluster['total'] + 1  
				cont += 1

for i in range(n_clusters):
	cluster = lista[i]
	if cluster['total'] != 0:
		print("================================cluster " + str(i + 1) +"==================================")
		print("atendimento: " + "%0.2f"%(cluster['atendimento']/cluster['total']*100) +"%")
		print("telecomunicacoes: " + "%0.2f"%(cluster['telecomunicacoes']/cluster['total']*100) +"%")
		print("saude: " + "%0.2f"%(cluster['saude']/cluster['total']*100) +"%")
		print("educacao_publica: " + "%0.2f"%(cluster['educacao_publica']/cluster['total']*100) +"%")
		print("infraestrutura: " + "%0.2f"%(cluster['infraestrutura']/cluster['total']*100) +"%")
		print("meio_ambiente: " + "%0.2f"%(cluster['meio_ambiente']/cluster['total']*100) +"%")
		print("transporte: " + "%0.2f"%(cluster['transporte']/cluster['total']*100) +"%")
		print("governamental: " + "%0.2f"%(cluster['governamental']/cluster['total']*100) +"%")
		print("total: " + str(cluster['total']) + " textos")
		print("\n")
	else:
		print("================================cluster " + str(i + 1) +"==================================")
		print("atendimento: 0%")
		print("telecomunicacoes: 0%")
		print("saude: 0%")
		print("educacao_publica: 0%")
		print("infraestrutura: 0%")
		print("meio_ambiente: 0%")
		print("transporte: 0%")
		print("governamental: 0%")
		print("total: " + cluster['total'] + " textos")
		print("\n")