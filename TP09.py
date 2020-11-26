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

for elem in items_rests:
    sentence = elem.getElementsByTagName("text")[0]
    aspect_terms = elem.getElementsByTagName("aspectTerm")
    print("//// Sentence : ////")
    print(sentence.firstChild.data)
    print("//// Terms : ////")
    for term in aspect_terms:
        print(term.getAttribute("term"))
