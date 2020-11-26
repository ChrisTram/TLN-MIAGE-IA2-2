import gensim
import nltk
from nltk.corpus import brown

model =gensim.models.Word2Vec(brown.sents())
model.save('brown.embedding') 
new_model = gensim.models.Word2Vec.load('brown.embedding')

print(len(new_model['university']))
print(new_model.similarity('university','school') > 0.3)

from nltk.data import find
word2vec_sample =str(find('models/word2vec_sample/pruned.word2vec.txt'))

model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False) 

print(len(model.vocab))
print(len(model['university']))

model.most_similar(positive=['university'], topn = 3)
model.most_similar(positive=['woman','king'], negative=['man'], topn = 1) 
model.most_similar(positive=['Paris','Germany'], negative=['Berlin'], topn = 1)