import nltk
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import unicodedata


def strip_accents(text):
    text = text.lower()
    try:
        text = encode(text,'utf-8')
    except (TypeError, NameError) as e:
        print(e)

    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode('utf-8')
    return str(text)    

class Preprocessor():
    

    def preprocess(string):
        s = string.strip()
        s = strip_accents(s)  
        s = re.sub('[^a-zA-Z]', ' ', s)
        return s


    def returnTFIDF(self, data):
        stopwordz = nltk.corpus.stopwords.words('portuguese')
        TV = TfidfVectorizer(analyzer="word",preprocessor=self.preprocess,stop_words=stopwordz, lowercase = True)
        TV2 = TV.fit_transform(text for text in data)
        print(TV.get_feature_names())

    def get_corpus():
        data = []

        

        for d in os.listdir("../../CidadaoData/2017/Dezembro"):
            dict = pickle.load(open("../../CidadaoData/2017/Dezembro/"+d,"rb"), encoding="utf-8")
            data.append(dict["reclamacao"].strip().lower())
        return data


data = Preprocessor.get_corpus()
vectorizer = TfidfVectorizer        
Preprocessor.returnTFIDF(Preprocessor, data)
   