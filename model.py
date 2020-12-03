import tensorflow as tf
import TP09
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

if __name__ == "__main__":
    df = TP09.get_datas('.\Dataset\Restaurants_Train.xml')
    df, label = TP09.data_pre_treatment(df)

    text = df['Sentences'].values
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(text)
    vocab_size = len(tokenizer.word_index) + 1
    encoded_docs = tokenizer.texts_to_sequences(text)
    padded_sequence = pad_sequences(encoded_docs, maxlen=200)

    print(tokenizer.word_index)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.layers import SpatialDropout1D
    from tensorflow.keras.layers import Embedding

    embedding_vector_length = 64

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

    history = model.fit(padded_sequence, label[0],
                        validation_split=0.2, epochs=5, batch_size=32)
