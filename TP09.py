import gensim
import nltk
import numpy as np
from nltk.corpus import brown
from nltk.data import find
from xml.dom import minidom
from text_processing import *

import pandas as pd


def get_datas(xmlPath):
    sentences = []
    terms = []
    polarities = []
    fromList = []
    toList = []
    doc = minidom.parse(xmlPath)
    items = doc.getElementsByTagName('sentence')
    for elem in items:
        sentence = elem.getElementsByTagName("text")[0]
        aspect_terms = elem.getElementsByTagName("aspectTerm")

        # We only keep datas with terms
        if len(aspect_terms) > 0:
            sentences.append(sentence.firstChild.data)
            sentence_terms = []
            sentence_polarities = []
            sentence_fromList = []
            sentence_toList = []
            for term in aspect_terms:
                sentence_terms.append(term.getAttribute("term"))
                sentence_polarities.append(term.getAttribute("polarity"))
                sentence_fromList.append(term.getAttribute("from"))
                sentence_toList.append(term.getAttribute("to"))

            terms.append(sentence_terms)
            polarities.append(sentence_polarities)
            fromList.append(sentence_fromList)
            toList.append(sentence_toList)

            # Create a Dataframe from all lists
            df = pd.DataFrame(sentences, columns=['Sentences'])
            df['Terms'] = terms
            df['Polarities'] = polarities
            df['From'] = fromList
            df['To'] = toList

    return df


df_rest = get_datas('.\Dataset\Restaurants_Train.xml')
df_lap = get_datas('.\Dataset\Laptop_Train.xml')



print(df_rest)
