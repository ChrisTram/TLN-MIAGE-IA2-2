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
    doc = minidom.parse(xmlPath)
    items = doc.getElementsByTagName('sentence')
    for elem in items:
        sentence = elem.getElementsByTagName("text")[0]
        aspect_terms = elem.getElementsByTagName("aspectTerm")
        #We only keep datas with terms
        if(len(aspect_terms) != 0):
            sentences.append(sentence.firstChild.data)
            sentence_terms = []
            for term in aspect_terms:
                sentence_terms.append(term.getAttribute("term"))
            terms.append(sentence_terms)
    return sentences, terms


sentences_rests, terms_rests = get_datas('.\Restaurants_Train.xml')
sentences_lapts, terms_lapts = get_datas('.\Laptop_Train.xml')


for i in range(len(sentences_rests)):
    print("//// Sentence Restaurant : ", sentences_rests[i])
    print("//// Terms : ", terms_rests[i])

for i in range(len(sentences_lapts)):
    print("//// Sentence Laptop : ", sentences_lapts[i])
    print("//// Terms : ", terms_lapts[i])




ps = PorterStemmer()
words_data = ['this','movie','is','wonderful']


pos_val = pos_tag(words_data)
senti_val = [get_sentiment(x,y) for (x,y) in pos_val]

print(f"pos_val is {pos_val}")
print(f"senti_val is {senti_val}")