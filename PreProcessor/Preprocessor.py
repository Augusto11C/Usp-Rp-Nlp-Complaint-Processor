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
        #print(e)
        pass

    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode('utf-8')
    return str(text)    

class Preprocessor():
    
    @classmethod
    def preprocess(cls,string):
        s = string.strip()
        s = strip_accents(s)  
        s = re.sub('[^a-zA-Z]', ' ', s)

        return s

    @classmethod
    def tokenize_string(cls,string):
        stopwordz = nltk.corpus.stopwords.words('portuguese')+["","\r\n","pois","que","pra","ter","fazer","ser","para"]
        words = string.split(" ")
        tokenizedWords = []
        for word in words:
            if(word not in stopwordz and not len(word) <= 2):
                tokenizedWords.append(word)
        return tokenizedWords

    @classmethod
    def returnTF(cls, data):
        stopwordz = nltk.corpus.stopwords.words('portuguese')
        TV = CountVectorizer(analyzer="word",preprocessor=Preprocessor.preprocess,stop_words=stopwordz, lowercase = True, max_features=1000)
        td = TV.fit_transform(text for text in data)
        #print(TV.get_feature_names())
        return td

    @classmethod
    def returnTFIDF(cls, data):
        stopwordz = nltk.corpus.stopwords.words('portuguese')
        TV = TfidfVectorizer(analyzer="word",preprocessor=Preprocessor.preprocess,stop_words=stopwordz, lowercase = True,max_features=1000)
        td = TV.fit_transform(text for text in data)
        #print(TV.get_feature_names())
        return td

    '''def get_corpus():
        data = []

        

        for d in os.listdir("../../CidadaoData/2017/Dezembro"):
            dict = pickle.load(open("../../CidadaoData/2017/Dezembro/"+d,"rb"), encoding="utf-8")
            data.append(dict["reclamacao"].strip().lower())
        return data
'''

'''data = Preprocessor.get_corpus()
vectorizer = TfidfVectorizer        
Preprocessor.returnTFIDF(Preprocessor, data)
'''