import gensim
import nltk
import numpy as np
from nltk.corpus import brown
from nltk.data import find
from xml.dom import minidom
from text_processing import *


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
        if len(aspect_terms) != 0:
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
    return sentences, terms, polarities, fromList, toList


sentences_rests, terms_rests, polarities_rest, fromList_rest, toList_rest = get_datas('.\Dataset\Restaurants_Train.xml')
sentences_lapts, terms_lapts, polarities_lapts, fromList_lapts, toList_lapts  = get_datas('.\Dataset\Laptop_Train.xml')

for i in range(len(sentences_rests)):
    print("//// Sentence Restaurant : ", sentences_rests[i])
    print("//// Terms : ", terms_rests[i])
    print("//// Polarities : ", polarities_rest[i])
    print("//// fromList : ", fromList_rest[i])
    print("//// toList : ", toList_rest[i])

for i in range(len(sentences_lapts)):
    print("//// Sentence Laptop : ", sentences_lapts[i])
    print("//// Terms : ", terms_lapts[i])
    print("//// Polarities : ", polarities_lapts[i])
    print("//// fromList : ", fromList_lapts[i])
    print("//// toList : ", toList_lapts[i])

