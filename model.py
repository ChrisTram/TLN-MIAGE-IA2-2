import tensorflow as tf
import TP09


if __name__ == "__main__":
    df = TP09.get_datas('.\Dataset\Restaurants_Train.xml')

    VOCAB_SIZE=1000
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])