import os
import pickle
from pprint import pprint
import re
import unicodedata
import numpy as np

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

def save_waifu(waifu, waifu_addr):
    waifu_dir = "/".join(waifu_addr.split('/')[:-1])
    if not os.path.exists(waifu_dir):
            os.makedirs(waifu_dir)
    with open(waifu_addr, "wb") as f:
        pickle.dump(waifu, f)    
    pass

class IncrementalDict(dict):
    def __missing__(self, key):
        return None

RESULTS_FILENAME = "TopicosResumidos.txt"
BASEFOLDER = 'CidadaoData'
SAVEFOLDER = "CategorizedWaifus/"
TOPICS = IncrementalDict()

atendimento = np.array([
    r'atendimento',
    r'comunicacao.*usuario',
    r'^denuncia.*',
    r'^servicos.*'
])

telecomunicacoes = np.array([
    r'banda.*larga.*fixa',
    r'tv.*por.*assinatura',
    r'telefonia'
])

saude = np.array([
    r'ambulatorios',
    r'hospitais',
    r'postos.*de.*saude',
    r'medicamentos',
    r'prestacao',
    r'zoonoses'
])

educacao_publica = np.array([
    r'escola',
    r'faculdades',
    r'creches'
])

infraestrutura = np.array([
    r'aeroporto',
    r'buracos',
    r'estradas',
    r'expansao',
    r'iluminacao',
    r'obras'
])

meio_ambiente = np.array([
    r'arvores',
    r'limpeza',
    r'lixo',
    r'meio',
    r'poluicao'
])

transporte = np.array([
    r'acessibilidade',
    r'bilhetagem',
    r'concessoes',
    r'terminais',
    r'transporte',
    r'transito',
    r'voos'
])

governamental = np.array([
    r'cidades',
    r'estados',
    r'^estadual$',
    r'uniao',
    r'cultura',
    r'seguranca',
    r'contratos',
    r'empregados',
    r'oportunidade',
    r'trabalho',
    r'cobranca',
    r'impostos',
    r'mensalidades'
])

def create_waifus():    
    for year in os.listdir(BASEFOLDER):
        if "waifu" in year:
            continue
        for month in os.listdir(BASEFOLDER + "/" + year):
            for waifu_name in os.listdir(BASEFOLDER + "/" + year + "/" + month):
                with open(BASEFOLDER + "/" + year + "/" + month + "/" + waifu_name, mode='rb') as waifu_file:
                    mitsuketa = False
                    waifu = pickle.load(waifu_file)
                    topic = waifu['categoria'].strip().split("/")[0].strip()
                    topic = text_to_id(topic).lower()

                    for raw_string in atendimento:
                        if re.search(raw_string, topic):
                            waifu['categoria'] = "atendimento"
                            mitsuketa = True                            
                        pass
                    if mitsuketa:
                        waifu_addr = SAVEFOLDER + year + '/' + month + '/' + waifu_name
                        save_waifu(waifu, waifu_addr)
                        continue

                    for raw_string in telecomunicacoes:
                        if re.search(raw_string, topic):
                            waifu['categoria'] = "telecomunicacoes"
                            mitsuketa = True                            
                        pass
                    if mitsuketa:
                        waifu_addr = SAVEFOLDER + year + '/' + month + '/' + waifu_name
                        save_waifu(waifu, waifu_addr)
                        continue
                        
                    for raw_string in saude:
                        if re.search(raw_string, topic):
                            waifu['categoria'] = "saude"
                            mitsuketa = True                            
                        pass
                    if mitsuketa:
                        waifu_addr = SAVEFOLDER + year + '/' + month + '/' + waifu_name
                        save_waifu(waifu, waifu_addr)
                        continue

                    for raw_string in educacao_publica:
                        if re.search(raw_string, topic):
                            waifu['categoria'] = "educacao_publica"
                            mitsuketa = True                            
                        pass
                    if mitsuketa:
                        waifu_addr = SAVEFOLDER + year + '/' + month + '/' + waifu_name
                        save_waifu(waifu, waifu_addr)
                        continue

                    for raw_string in infraestrutura:
                        if re.search(raw_string, topic):
                            waifu['categoria'] = "infraestrutura"
                            mitsuketa = True                            
                        pass
                    if mitsuketa:
                        waifu_addr = SAVEFOLDER + year + '/' + month + '/' + waifu_name
                        save_waifu(waifu, waifu_addr)
                        continue

                    for raw_string in meio_ambiente:
                        if re.search(raw_string, topic):
                            waifu['categoria'] = "meio_ambiente"
                            mitsuketa = True                            
                        pass
                    if mitsuketa:
                        waifu_addr = SAVEFOLDER + year + '/' + month + '/' + waifu_name
                        save_waifu(waifu, waifu_addr)
                        continue

                    for raw_string in transporte:
                        if re.search(raw_string, topic):
                            waifu['categoria'] = "transporte"
                            mitsuketa = True                            
                        pass
                    if mitsuketa:
                        waifu_addr = SAVEFOLDER + year + '/' + month + '/' + waifu_name
                        save_waifu(waifu, waifu_addr)
                        continue

                    for raw_string in governamental:
                        if re.search(raw_string, topic):
                            waifu['categoria'] = "governamental"
                            mitsuketa = True                            
                        pass
                    if mitsuketa:
                        waifu_addr = SAVEFOLDER + year + '/' + month + '/' + waifu_name
                        save_waifu(waifu, waifu_addr)
                        continue


def confirm_waifus():
    count_dick = IncrementalDict()
    for year in os.listdir(SAVEFOLDER):
        for month in os.listdir(SAVEFOLDER + "/" + year):
            for waifu_name in os.listdir(SAVEFOLDER + "/" + year + "/" + month):
                with open(SAVEFOLDER + "/" + year + "/" + month + "/" + waifu_name, mode='rb') as waifu_file:
                    waifu = pickle.load(waifu_file)
                    waifu_cat = waifu['categoria']
                    if not count_dick[waifu_cat]:
                        count_dick[waifu_cat] = 1
                    else:
                        count_dick[waifu_cat] += 1
    pprint(count_dick)

import sys
if __name__ == "__main__":
    option = sys.argv[1]
    if option == "create":
        create_waifus()
        print("Waifus categorized and created anew successfully")
    
    elif option == "confirm":
        confirm_waifus()
        
    pass    

