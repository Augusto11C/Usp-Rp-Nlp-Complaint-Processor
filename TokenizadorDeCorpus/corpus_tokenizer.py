import nltk
import pickle
import os
import numpy as np
from sklearn import metrics
import sys
import re
from sklearn.cluster import KMeans

sys.path.insert(1, '..\Preprocessor')
from Preprocessor import Preprocessor


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

def tokenizar(data):
    data_tokenized = []
    preprocessor = Preprocessor
    i = 1
    for text in data:
        print("Tokenizando texto " + str(i))
        preprocessed_text = preprocessor.preprocess(text)
        words_list = preprocessor.tokenize_string(preprocessed_text)
        data_tokenized.append(words_list)
        i = i+1 
    pickle.dump(data_tokenized,open("corpus_tokenizado_sem_java.lai","wb"))

resultados_kmeans = pickle.load(open('resultados_kmeans_sem_java.jojo','rb'))
data = resultados_kmeans['data']
tokenizar(data)