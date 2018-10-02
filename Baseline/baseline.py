import nltk
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from sklearn.metrics import silhouette_score
import sys
sys.path.insert(0, '../CloudCode')
import wordcloud
import PalavrasMaisFrequentesPorCluster
import GroupedColorFunc

data = []

stopwordz = nltk.corpus.stopwords.words('portuguese')

for d in os.listdir("../../CidadaoData/2017/Dezembro"):
	dict = pickle.load(open("../../CidadaoData/2017/Dezembro/"+d,"rb"))
	data.append(dict["reclamacao"].strip())

CV = CountVectorizer(analyzer="word",preprocessor=None,stop_words=stopwordz) 

td = CV.fit_transform(text for text in data)

kmeans = KMeans(n_clusters=2, verbose=1).fit(td)

TV = TfidfVectorizer(analyzer="word",preprocessor=None,stop_words=stopwordz)

td = TV.fit_transform(text for text in data)

kmeans = KMeans(n_clusters=2, verbose=1).fit(td)

labels = kmeans.labels_
print(metrics.silhouette_score(td, labels, metric='euclidean'))



