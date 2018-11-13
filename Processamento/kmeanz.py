import nltk
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from sklearn.metrics import silhouette_score
import sys
import re

sys.path.insert(0, '..\CloudCode')
from Ourwordcloud import Ourwordcloud
sys.path.insert(1, '..\Preprocessor')
from Preprocessor import Preprocessor
# import Ourwordcloud
from PalavrasMaisFrequentesPorCluster import PalavrasMaisFrequentesPorCluster
import GroupedColorFunc


stopwordz = nltk.corpus.stopwords.words('portuguese')
preprocessor = Preprocessor

def load_waifus(dump=True):
    print("loading database")
    data = []
    for year in os.listdir("../../CidadaoData"):
        if os.path.isdir("../../CidadaoData/"+year):
            for month in os.listdir("../../CidadaoData/"+year):
                if os.path.isdir("../../CidadaoData/"+year+"/"+month):
                    for waifu in os.listdir("../../CidadaoData/"+year+"/"+month):
                        with open("../../CidadaoData/"+year+"/"+month+"/"+waifu,"rb") as f:
                            data.append(pickle.load(f, encoding="utf-8")["reclamacao"])
    data = tuple(data)
    if dump:
        with open("data.tuple", "wb") as f:
            pickle.dump(data, f)
            return data

    return data

data_is_present = False
for file in os.listdir("."):
    if re.search(r"data.tuple", file):
        data_is_present = True
        break

if data_is_present:
    with open("data.tuple", "rb") as f:
        data = pickle.load(f)
else:
    data = load_waifus()

print("converting to numeric matrix")
td = preprocessor.returnTF(data)
print(str(td.shape))
kmeanz_list = []
scorez = []
for i in range(5,12):
    print("evaluating "+str(i)+ " clusters")
    kmeans = KMeans(n_clusters=i,n_jobs=-1,verbose=1).fit(td)
    kmeanz_list.append(kmeans)
    labels = kmeans.labels_
    scorrre=metrics.silhouette_score(td, labels, metric='euclidean',sample_size =15000)
    print("the score is "+str(scorrre))
    scorez.append(scorrre)

dictout={}
dictout["data"]=data
dictout["kmeans"]=kmeanz_list
dictout["scores"]= scorez

pickle.dump(dictout,open("resultados_kmeans.jojo","wb"))