
import nltk
import os
import pickle


class LoadCorpus():

    def get_corpus():
            data = []

            stopwordz = nltk.corpus.stopwords.words('portuguese')

            for d in os.listdir("../../CidadaoData/2017/Dezembro"):
                dict = pickle.load(open("../../CidadaoData/2017/Dezembro/"+d,"rb"), encoding="utf-8")
                data.append(dict["reclamacao"].strip().lower())
            return data