import tensorflow as tf
import TP09
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def append_feature_to_vector(vector, feature):
    features = []

    for i in range(len(vector)):
        a = vector[i]
        a = np.append(a, feature[i])

        features.append(a)

    return features


if __name__ == "__main__":
    df = TP09.get_datas('.\Dataset\Restaurants_Train.xml')
    df, label = TP09.data_pre_treatment(df)

    maxlen = 100

    # Sentences to bag of words
    text = df['Sentences'].values
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(text)
    vocab_size = len(tokenizer.word_index) + 1
    encoded_docs = tokenizer.texts_to_sequences(text)
    text_padded_sequence = pad_sequences(encoded_docs, maxlen=maxlen)

    # Terms to bag of words, we use the same tokenizer we got from the sentences
    terms = df['Terms'].values
    encoded_docs = tokenizer.texts_to_sequences(terms)
    terms_padded_sequence = pad_sequences(encoded_docs, maxlen=10)

    print(tokenizer.word_index)

    # Append the different features to the bag of words
    features = append_feature_to_vector(text_padded_sequence, terms_padded_sequence)
    features = append_feature_to_vector(features, df['Sentiments_Scores'].to_numpy())

    print(features)
    features = np.array(features)

    embedding_vector_length = 100

    # We define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_vector_length, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(features, label[0],
                        validation_split=0.2, epochs=15, batch_size=32)
