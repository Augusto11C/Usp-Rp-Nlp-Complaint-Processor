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

import nltk

n_samples = 2000
n_features = 3000
n_components = 10
n_top_words = 20

BASEFOLDER = '../../CategorizedWaifus/'


def load_waifus(dump=False):
    waifu_list = []
    waifu_path_list = []
    for year in os.listdir(BASEFOLDER):
        # if "waifu" in year:
        #     continue
        for month in os.listdir(BASEFOLDER + "/" + year):
            for waifu_name in os.listdir(BASEFOLDER + "/" + year + "/" + month):
                with open(BASEFOLDER + "/" + year + "/" + month + "/" + waifu_name, 'rb') as waifu_file:
                    waifu = pickle.load(waifu_file)
                    waifu_list.append(waifu['reclamacao'].strip())
                    waifu_path_list.append( year + "/" + month + "/" + waifu_name)

    waifu_list = tuple(waifu_list)
    waifu_path_list = tuple(waifu_path_list)

    if dump:
        with open("waifu_list.tuple", "wb") as f:
            pickle.dump(waifu_list,f)
        with open("waifu_path_list.tuple", "wb") as f:
            pickle.dump(waifu_path_list,f)
    
    return waifu_list, waifu_path_list


if __name__ == "__main__":
    waifu_alr_exists = False
    for file_name in os.listdir('.'):
        if re.search("waifu", file_name):
            waifu_alr_exists = True
            print("Waifus already saved; using existing ones")
    if waifu_alr_exists:
        with open("waifu_list.tuple", "rb") as f:
            waifu_list = pickle.load(f)
        with open("waifu_path_list.tuple", "rb") as f:
            waifu_path_list = pickle.load(f)
    else:
        waifu_list, waifu_path_list = load_waifus(dump=True)


    lda_from_waifus(waifu_list, 6)

    

    pass

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def lda_from_waifus(waifu_list: tuple, n_topics=8):      
    print("Loading dataset...")
    t0 = time()
    # Using tf-idf features for LDA
    stopwords = nltk.corpus.stopwords.words('portuguese')+["","\r\n","pois","que","pra","ter","fazer","ser","para"]
    print("Extracting tf-idf features from the waifus...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95,
                                       max_features=n_features,
                                       stop_words=stopwords)

    tfidf = tfidf_vectorizer.fit_transform(waifu for waifu in waifu_list)
    print("done in %0.3fs." % (time() - t0))
    
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
    tfidf_featurte_names = tfidf_vectorizer.get_feature_names()#matriz, linha palavra coluna documento
    print("Fitting LDA models with tf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    t0 = time()
    lda.fit(tfidf)
    print("done in %0.3fs." % (time() - t0))
    print("\nTopics in LDA model:")
    print_top_words(lda, tfidf_featurte_names, 5)

