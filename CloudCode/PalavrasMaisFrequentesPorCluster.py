import nltk
import pickle
import os
from sklearn.cluster import KMeans


class PalavrasMaisFrequentesCluster:

	def __get_corpus():
    data = []

	stopwordz = nltk.corpus.stopwords.words('portuguese')

	for d in os.listdir("../../CidadaoData/2017/Dezembro"):
		dict = pickle.load(open("../../CidadaoData/2017/Dezembro/"+d,"rb"))
		data.append(dict["reclamacao"].strip())
	return data