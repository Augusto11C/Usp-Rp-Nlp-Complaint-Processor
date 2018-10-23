import os
import pickle
from pprint import pprint
import re
import unicodedata

def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def text_to_id(text):
    """
    Convert input text to id.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    text = strip_accents(text.lower())
    text = re.sub('[^0-9a-zA-Z]', ' ', text)
    return text


class IncrementalDict(dict):
    def __missing__(self, key):
        return None

RESULTS_FILENAME = "TopicosResumidos.txt"
BASEFOLDER = 'CidadaoData'
# PPRINTER = PrettyPrinter(4)

TOPICS = IncrementalDict()

atendimento = [
    r'atendimento',
    r'comunicacao.*usuario',
    r'^denuncia.*',
    r'^servicos.*'
]

telecomunicacoes = [
    r'banda.*larga.*fixa',
    r'tv.por.assinatura',
    r''
]

if __name__ == "__main__":
    for year in os.listdir(BASEFOLDER):
        if "waifu" in year:
            continue
        for month in os.listdir(BASEFOLDER + "/" + year):
            for waifu_name in os.listdir(BASEFOLDER + "/" + year + "/" + month):
                with open(BASEFOLDER + "/" + year + "/" + month + "/" + waifu_name, mode='rb') as waifu_file:
                    waifu = pickle.load(waifu_file)
                    topic = waifu['categoria'].strip().split("/")[0].strip().lower()
                    for raw_string in atendimento:
                        if re.search(raw_string, topic):
                            waifu['categoria'] = "Atendimento"
                        
                        pass
                    

    with open(RESULTS_FILENAME, "w", encoding='UTF-8') as results_file:
        pprint(TOPICS, results_file)
        results_file.write("No. of Topics: {};".format(len(TOPICS.keys())))

    pass