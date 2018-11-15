"""
=======================================================================================
Topic extraction with Non-negative Matrix Factorization and Latent Dirichlet Allocation
=======================================================================================

This is an example of applying :class:`sklearn.decomposition.NMF` and
:class:`sklearn.decomposition.LatentDirichletAllocation` on a corpus
of documents and extract additive models of the topic structure of the
corpus.  The output is a list of topics, each represented as a list of
terms (weights are not shown).

Non-negative Matrix Factorization is applied with two different objective
functions: the Frobenius norm, and the generalized Kullback-Leibler divergence.
The latter is equivalent to Probabilistic Latent Semantic Indexing.

The default parameters (n_samples / n_features / n_components) should make
the example runnable in a couple of tens of seconds. You can try to
increase the dimensions of the problem, but be aware that the time
complexity is polynomial in NMF. In LDA, the time complexity is
proportional to (n_samples * iterations).

"""

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pickle, os
import re
import sys
import nltk
import numpy as np


BASEWAIFUSFOLDER = '../../CategorizedWaifus/'

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #{}:".format(topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print("")

def load_waifus(dump=False, use_word2vec=False): #returns tuples
    waifu_list = []
    waifu_path_list = []
    for year in os.listdir(BASEWAIFUSFOLDER):
        # if "waifu" in year:
        #     continue
        for month in os.listdir(BASEWAIFUSFOLDER + "/" + year):
            for waifu_name in os.listdir(BASEWAIFUSFOLDER + "/" + year + "/" + month):
                with open(BASEWAIFUSFOLDER + "/" + year + "/" + month + "/" + waifu_name, 'rb') as waifu_file:
                    waifu = pickle.load(waifu_file, encoding="utf-8")
                    if not use_word2vec:
                        waifu_list.append(waifu['reclamacao'].strip())
                    else:
                        try:
                            waifu_list.append(waifu['word2vec'])
                        except:
                            print("Your waifu database does not contain word embeddings!")
                            sys.exit(-1)
                    waifu_path_list.append( year + "/" + month + "/" + waifu_name)
                            
                    

    waifu_list = tuple(waifu_list)
    waifu_path_list = tuple(waifu_path_list)

    if dump:
        if use_word2vec:
            with open("waifu_w2v.tuple", "wb") as f:
                pickle.dump(waifu_list,f)    
        else:
            with open("waifu_list.tuple", "wb") as f:
                pickle.dump(waifu_list,f)
        with open("waifu_path_list.tuple", "wb") as f:
            pickle.dump(waifu_path_list,f)
    
    return waifu_list, waifu_path_list


    # Use tf (raw term count) features for LDA.
    # print("Extracting tf features for LDA...")
    # tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
    #                                 max_features=n_features,
    #                                 stop_words='english')
    # t0 = time()
    # tf = tf_vectorizer.fit_transform(data_samples)
    # print("done in %0.3fs." % (time() - t0))
    # Fit the NMF model
    # print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
    #       "n_samples=%d and n_features=%d..."
    #       % (n_samples, n_features))
    # t0 = time()
    # nmf = NMF(n_components=n_components, random_state=1,
    #           alpha=.1, l1_ratio=.5).fit(tfidf)
    # print("done in %0.3fs." % (time() - t0))
    # Fit the NMF model
    # print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
    #       "tf-idf features, n_samples=%d and n_features=%d..."
    #       % (n_samples, n_features))
    #     # t0 = time()
    # nmf = NMF(n_components=n_components, random_state=1,
    #           beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
    #           l1_ratio=.5).fit(tfidf)
    # print("done in %0.3fs." % (time() - t0))

def lda_from_waifus(waifus: tuple, n_topics=8, n_features=1000, n_top_words=20, use_word2vec=False):      
    print("Loading dataset...")
    print("Dataset length: {}".format(len(waifus)))
    
    stopwords = nltk.corpus.stopwords.words('portuguese') + ["","\r\n","pois","que","pra","ter","fazer","ser","para"]
    t0 = time()
    
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,    
                                    random_state=0, n_jobs=-1)
    
    if not use_word2vec:
        # Using tf-idf features for LDA
        print("Extracting tf-idf features from the waifus...")
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                           max_features=n_features,
                                           stop_words=stopwords)

        tfidf = tfidf_vectorizer.fit_transform(waifu for waifu in waifus)
        print("done in %0.3fs." % (time() - t0))
    
        tfidf_feature_names = tfidf_vectorizer.get_feature_names() 
        
        t0 = time()
        print("Fitting LDA models with tfidf features, "
              "n_features=%d..."
            % n_features)
            
        lda.fit(tfidf)
        print("done in %0.3fs." % (time() - t0))
        print("\nTopics in LDA model:")
        print_top_words(lda, tfidf_feature_names, n_top_words)
        return lda, tfidf_feature_names
    
    else:
        print("Extracting word embeddings from the waifus...")
        w2v_matrix = np.stack(waifus)
        print("Fitting LDA models with word embeddings, "
              "n_features=%d..."
            % len(waifus[0]))
        lda.fit(w2v_matrix)
        print("done in %0.3fs." % (time() - t0))
        return lda
        


if __name__ == "__main__":  
    n_features = 500
    n_topics = int(sys.argv[1])
    use_word2vec = sys.argv([2]).lower() in ("1", "true") 
    n_top_words = 20
    waifu_alr_exists = False
    
    for file_name in os.listdir('.'):
        if re.search("waifu", file_name):
            waifu_alr_exists = True
            print("waifus found, using existing ones")
            break;
    if waifu_alr_exists:
        with open("waifu_list.tuple", "rb") as f:
            waifu_list = pickle.load(f)
        with open("waifu_path_list.tuple", "rb") as f:
            waifu_path_list = pickle.load(f)
    else:
        waifu_list, waifu_path_list = load_waifus(dump=True)

    lda = lda_from_waifus(waifu_list, n_topics=n_topics, n_features=n_features, n_top_words=n_top_words)
    with open('LDA_{}_OBJ_{}.lda'.format(n_features ,n_topics), "wb") as f:
        pickle.dump(lda, f)


    pass
