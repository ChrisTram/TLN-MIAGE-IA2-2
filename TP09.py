from enum import Enum

import gensim
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import brown
from nltk.data import find
from xml.dom import minidom
from text_processing import *



def get_datas(xmlPath):
    sentences = []
    terms = []
    polarity = []
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
                # Add all features to the lists
                # For each term we had the full sentence as a feature
                sentences.append(sentence.firstChild.data)
                terms.append(term.getAttribute("term"))
                polarity.append(term.getAttribute("polarity"))
                sents_scores.append(get_sentiment_score(term.getAttribute("term")))
                fromList.append(term.getAttribute("from"))
                toList.append(term.getAttribute("to"))

            # Create a Dataframe from all lists
            df = pd.DataFrame(sentences, columns=['Sentences'])
            df['Terms'] = terms
            df['Sentiments_Scores'] = sents_scores
            df['polarity'] = polarity
            df['From'] = fromList
            df['To'] = toList

    return df


def data_pre_treatment(df):
    # We remove the term with 'conflict' label
    df = df[df['polarity'] != 'conflict']

    # convert polarity to numeric
    label = df['polarity'].factorize()

    # We dont need the label in the dataframe anymore
    df.drop(['polarity'], axis=1)

    return df, label


if __name__ == "__main__":
    # df_rest = get_datas('.\Dataset\Restaurants_Test_Gold.xml')
    df_rest = get_datas('.\Dataset\Restaurants_Train.xml')
    df_lap = get_datas('.\Dataset\Laptop_Train.xml')

    df_rest, label = data_pre_treatment(df_rest)
    print(df_rest)
    print(label)
