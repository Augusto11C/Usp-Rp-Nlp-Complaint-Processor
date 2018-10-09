import os
import pickle
from pprint import pprint

class IncrementalDict(dict):
    def __missing__(self, key):
        return None

RESULTS_FILENAME = "TopicosResumidos.txt"
BASEFOLDER = 'CidadaoData'
# PPRINTER = PrettyPrinter(4)

TOPICS = IncrementalDict()

if __name__ == "__main__":
    for year in os.listdir(BASEFOLDER):
        if "waifu" in year:
            continue
        for month in os.listdir(BASEFOLDER + "/" + year):
            for waifu_name in os.listdir(BASEFOLDER + "/" + year + "/" + month):
                with open(BASEFOLDER + "/" + year + "/" + month + "/" + waifu_name, 'rb') as waifu_file:
                    waifu = pickle.load(waifu_file)
                    topic = waifu['categoria'].strip().split("/")[0].strip().split()
                    key = topic[0]
                    topic_suffix = "".join(topic[1:])
                    if not TOPICS[key]:
                        TOPICS[key] = [set(), 0]
                    TOPICS[key][0].add(topic_suffix)
                    TOPICS[key][1] += 1

    with open(RESULTS_FILENAME, "w", encoding='UTF-8') as results_file:
        pprint(TOPICS, results_file)
        results_file.write("No. of Topics: {};".format(len(TOPICS.keys())))

    pass
