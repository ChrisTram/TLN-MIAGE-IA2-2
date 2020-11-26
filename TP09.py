import gensim
import nltk
import numpy as np
from nltk.corpus import brown
from nltk.data import find
from xml.dom import minidom

# parse an xml file by name
rests = minidom.parse('.\Restaurants_Train.xml')
laptops = minidom.parse('.\Laptop_Train.xml')

items_rests = rests.getElementsByTagName('sentences')

for elem in items_rests:
    print(elem)