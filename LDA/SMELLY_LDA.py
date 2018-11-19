from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
import pickle, os
import re
import sys
import numpy as np
import nltk

BASEWAIFUSFOLDER = '../CategorizedWaifus+W2VArrombz/'
WAIFU_LIST_PATH = "waifu_list.tuple"
WAIFU_W2V_PATH = "waifu_w2v.tuple"


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #{}:".format(topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print("")


def return_top_words(model, feature_names, n_top_words):
    top_words = []
    for topic in model.components_:
        top_words.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return top_words


def load_waifus(dump=False, use_word2vec=False):  # returns tuples
    waifu_list = []
    waifu_path_list = []
    for year in os.listdir(BASEWAIFUSFOLDER):
        # Only get waifus from 2017
        if not re.search(r"2017", year):
            continue
        for month in os.listdir(BASEWAIFUSFOLDER + "/" + year):
            for waifu_name in os.listdir(BASEWAIFUSFOLDER + "/" + year + "/" + month):
                with open(BASEWAIFUSFOLDER + "/" + year + "/" + month + "/" + waifu_name, 'rb') as waifu_file:
                    waifu = pickle.load(waifu_file)
                    if not use_word2vec:
                        waifu_list.append(waifu['reclamacao'].strip())
                    else:
                        try:
                            waifu_list.append(waifu['word2vec'])
                        except:
                            print("Your waifu database does not contain word embeddings!")
                            sys.exit(-1)
                    waifu_path_list.append(year + "/" + month + "/" + waifu_name)

    waifu_list = tuple(waifu_list)
    waifu_path_list = tuple(waifu_path_list)

    if dump:
        if use_word2vec:
            with open(WAIFU_W2V_PATH, "wb") as f:
                pickle.dump(waifu_list, f)
        else:
            with open(WAIFU_LIST_PATH, "wb") as f:
                pickle.dump(waifu_list, f)
        with open("waifu_path_list.tuple", "wb") as f:
            pickle.dump(waifu_path_list, f)

    return waifu_list, waifu_path_list


def lda_from_waifus(waifus: tuple, n_topics=8, n_features=1000, n_top_words=20, use_word2vec=False):
    print("Loading dataset...")
    print("Dataset length: {}".format(len(waifus)))

    try:
        stopwords = set(nltk.corpus.stopwords.words('portuguese') +
                        ["", "\r\n", "pois", "que", "pra", "ter", "fazer", "ser", "para"])
    except LookupError:
        print("No stopwords lists were found, downloading them now")
        nltk.download("stopwords")
        return lda_from_waifus(waifus, n_topics, n_features, n_top_words, use_word2vec)
    except UnicodeEncodeError as ue:
        print(ue)
        print("Well damnit.")
        sys.exit(-1)

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

        tfidf = tfidf_vectorizer.fit_transform(waifu for waifu in waifus).todense()
        print("done in %0.3fs." % (time() - t0))

        tfidf_feature_names = tfidf_vectorizer.get_feature_names()

        t0 = time()
        print("Fitting LDA models with tfidf features, "
              "n_features=%d..."
              % n_features)

        lda.fit(tfidf)
        print("done in %0.3fs." % (time() - t0))
        print("\nTopics in LDA model:")
        try:
            print_top_words(lda, tfidf_feature_names, n_top_words)
        except:
            print("Cant print lel")
        return lda, tfidf, return_top_words(lda, tfidf_feature_names, n_top_words)

    else:
        print("Extracting word embeddings from the waifus...")
        w2v_matrix = np.stack(waifus)
        print("Fitting LDA models with word embeddings, "
              "n_features=%d..."
              % len(waifus[0]))
        lda.fit(w2v_matrix)
        print("done in %0.3fs." % (time() - t0))
        return lda, w2v_matrix, None


def extract_clusters_from_lda(lda_obj: LatentDirichletAllocation, fitting_data):
    cluster_args = []
    # 1. normalize each of the vectors from both the lda matrix and the one used for fitting it
    fitting_data = normalize(fitting_data, norm='l1')
    lda_vectors = lda_obj.components_  # LDA is already l1 normalized
    n_clusters = lda_vectors.shape[0]
    if fitting_data.shape[1] != lda_vectors.shape[1]:
        print("Incompatible shapes: {} for lda and {} for the fitting data"
              .format(fitting_data.shape, lda_vectors.shape))
        sys.exit(-1)
    # 2. for each row of the fitting matrix do:
    for row in fitting_data:
        # 2.3. replicate it times the # of topics
        row_x_topics = np.tile(row, (n_clusters, 1))
        # 2.4. get the sum of the manhattan distances and
        #      get the arg for the min value from the resulting array,
        cluster_arg = np.sum(np.abs(row_x_topics - lda_vectors), axis=1).argmax()
        cluster_args.append(cluster_arg)

    print("Number of waifus taken into account: %d" % len(cluster_args))
    return tuple(cluster_args)


if __name__ == "__main__":
    n_topics = int(sys.argv[1])  # argument 1
    use_word2vec = sys.argv[2].lower() in ("1", "true")  # argument 2
    n_features = 1000
    mode = "REG"
    n_top_words = 20
    waifus_alr_exist = False
    w2v_waifus_exist = False

    for file_name in os.listdir('.'):
        if re.search(r"waifu_list", file_name):
            waifus_alr_exist = True
            print("waifu paths found, using existing ones")

        if re.search(r"waifu_w2v", file_name):
            w2v_waifus_exist = True
            print("w2v waifus found, using existing ones")

    if use_word2vec:
        mode = "W2V"
        if w2v_waifus_exist:
            with open(WAIFU_W2V_PATH, "rb") as f:
                waifu_list = pickle.load(f)
            with open("waifu_path_list.tuple", "rb") as f:
                waifu_path_list = pickle.load(f)
        else:
            waifu_list, waifu_path_list = load_waifus(dump=True, use_word2vec=True)
        lda, fitting_data, top_words = \
            lda_from_waifus(waifu_list, n_topics=n_topics, n_features=n_features, n_top_words=n_top_words, use_word2vec=True)
    else:
        if waifus_alr_exist:
            with open(WAIFU_LIST_PATH, "rb") as f:
                waifu_list = pickle.load(f)
            with open("waifu_path_list.tuple", "rb") as f:
                waifu_path_list = pickle.load(f)
        else:
            waifu_list, waifu_path_list = load_waifus(dump=True)
        lda, fitting_data, top_words = \
            lda_from_waifus(waifu_list, n_topics=n_topics, n_features=n_features, n_top_words=n_top_words)

    with open('LDA_{}_{}_{}.lda'.format(mode, n_features, n_topics), "wb") as f:
        pickle.dump(lda, f)
    if top_words:
        with open("TOP_WORDS_DECREASING_LDA_{}_{}_{}.list".format(mode, n_features, n_topics), "wb") as f:
            pickle.dump(top_words, f)

    # Extract the actual clusters from the resulting lda
    cluster_args = extract_clusters_from_lda(lda, fitting_data)
    with open('CLUSTER_ARGS_{}_{}_{}.tuple'.format(mode, n_features, n_topics), "wb") as f:
        pickle.dump(cluster_args, f)
    pass
