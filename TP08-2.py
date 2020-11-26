import numpy as np
import gensim
import nltk
from nltk.corpus import brown

from nltk.data import find
word2vec_sample =str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

model_v = model['university']
labels = model.most_similar(positive=[model_v], topn=50)

print(labels)


# It is recommended to use PCA first to reduce to ~50 dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_50 = pca.fit_transform(np.identity(50))
# Using TSNE to further reduce to 2 dimensions
from sklearn.manifold import TSNE
model_tsne = TSNE(n_components=2, random_state=0)
Y = model_tsne.fit_transform(X_50)


# Show the scatter plots
import matplotlib.pyplot as plt
plt.scatter(Y[:,0], Y[:,1], 20)
# Add labels
for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy = (x,y), xytext = (0, 0), textcoords = 'offset points', size = 10)
plt.show()