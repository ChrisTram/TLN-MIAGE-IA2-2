from enum import Enum

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
    sents_scores = []
    fromList = []
    toList = []

    doc = minidom.parse(xmlPath)
    items = doc.getElementsByTagName('sentence')
    for elem in items:
        sentence = elem.getElementsByTagName("text")[0]
        aspect_terms = elem.getElementsByTagName("aspectTerm")

        # We only keep datas with terms
        if len(aspect_terms) > 0:
            for term in aspect_terms:
                # We dont keep the term with 'conflict' label
                if term.getAttribute("polarity") != 'conflict':
                    # Add all features to the lists
                    # For each term we had the full sentence as a features
                    sentences.append(sentence.firstChild.data)
                    terms.append(term.getAttribute("term"))
                    polarities.append(term.getAttribute("polarity"))
                    sents_scores.append(get_sentiment_score(term.getAttribute("term")))
                    fromList.append(term.getAttribute("from"))
                    toList.append(term.getAttribute("to"))

            # Create a Dataframe from all lists
            df = pd.DataFrame(sentences, columns=['Sentences'])
            df['Terms'] = terms
            df['Sentiments_Scores'] = sents_scores
            df['Polarities'] = polarities
            df['From'] = fromList
            df['To'] = toList

    return df


def data_pre_treatment(df):
    # We remove
    label = df['Polarities']

    return label


if __name__ == "__main__":
    df_rest = get_datas('.\Dataset\Restaurants_Train.xml')
    df_lap = get_datas('.\Dataset\Laptop_Train.xml')

    print(df_rest)

    #print(data_pre_treatment(df_rest))
