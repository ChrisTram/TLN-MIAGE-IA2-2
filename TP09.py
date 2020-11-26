import gensim
import nltk
import numpy as np
from nltk.corpus import brown
from nltk.data import find
from xml.dom import minidom

# parse an xml file by name
rests = minidom.parse('.\Restaurants_Train.xml')
laptops = minidom.parse('.\Laptop_Train.xml')

items_rests = rests.getElementsByTagName('sentence')
items_lapts = rests.getElementsByTagName('sentence')

sentences_rests = []
terms_rests = []
sentences_lapts = []
terms_lapts = []



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


sentences_rest, terms_rests = get_datas('.\Restaurants_Train.xml')
sentences_lapts, terms_lapts = get_datas('.\Laptop_Train.xml')


for i in range(len(sentences_rests)):
    print("//// Sentence Restaurant : ", sentences_rests[i])
    print("//// Terms : ", terms_rests[i])

for i in range(len(sentences_lapts)):
    print("//// Sentence Restaurant : ", sentences_lapts[i])
    print("//// Terms : ", terms_lapts[i])