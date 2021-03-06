import numpy as np
import pandas as pd
from xml.dom import minidom
from text_processing import *
import tensorflow as tf

import model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_datas(xmlPath):
    sentences = []
    terms = []
    polarity = []
    sents_scores = []
    fromList = []
    toList = []

    doc = minidom.parse(xmlPath)
    items = doc.getElementsByTagName('sentence')
    for elem in items:
        sentence = elem.getElementsByTagName("text")[0]
        aspect_terms = elem.getElementsByTagName("aspectTerm")

        # We only keep datas with terms
        if len(aspect_terms) > 0:
            for term in aspect_terms:
                # Add all features to the lists
                # For each term we had the full sentence as a feature
                sentences.append(sentence.firstChild.data)
                terms.append(term.getAttribute("term"))
                polarity.append(term.getAttribute("polarity"))
                sents_scores.append(get_sentiment_score(term.getAttribute("term"), sentence.firstChild.data))
                fromList.append(term.getAttribute("from"))
                toList.append(term.getAttribute("to"))

            # Create a Dataframe from all lists
            df = pd.DataFrame(sentences, columns=['Sentences'])
            df['Terms'] = terms
            df['Sentiments_Scores'] = sents_scores
            df['Sentiments_Scores'] = df['Sentiments_Scores'].factorize()[0]
            df['polarity'] = polarity
            df['From'] = fromList
            df['To'] = toList

    return df


def data_pre_treatment(df):
    # We remove the term with 'conflict' label
    df = df[df['polarity'] != 'conflict']

    # convert polarity to numeric
    label = df['polarity'].factorize()

    # We dont need the label in the dataframe anymore
    # df.drop(['polarity'], axis=1)

    return df, label


def make_prediction_noLabel(df, saved_model):
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

    # Append the different features to the bag of words
    features = model.append_feature_to_vector(text_padded_sequence, terms_padded_sequence)
    features = model.append_feature_to_vector(features, df['Sentiments_Scores'].to_numpy())

    features = np.array(features)

    # Predict the entire dataset
    predictions = saved_model.predict(features)

    return predictions


if __name__ == "__main__":
    ######## Evaluation des fichier Restaurants
    path_r = 'Restaurants'

    # Load the dataframe without the label
    df_no_label = get_datas('.\Dataset\\' + path_r + '_Test_NoLabels.xml')
    df_no_label, _ = data_pre_treatment(df_no_label)

    # Load data with label
    df_Label = get_datas('.\Dataset\\' + path_r + '_Test_Gold.xml')
    _, label_gold = data_pre_treatment(df_Label)

    # Load the model
    saved_model = tf.keras.models.load_model(path_r + '_model.h5')

    # On fait la prediction sur les données du fichier Test_NoLabels
    r_predictions = make_prediction_noLabel(df_no_label, saved_model)

    # Evaluate the model
    r_metrics = saved_model.evaluate(r_predictions, label_gold[0])
    print("""
    //////////////////////////////////////////////////
    //////////// PREDICTION RESTAURANT ///////////////
    //////////////////////////////////////////////////
    """)
    print(str(saved_model.metrics_names[1]) + ': ' + str(r_metrics[1]))


    ######## Evaluation des fichier Laptop
    path_l = 'Laptop'

    # Load the dataframe without the label
    df_no_label = get_datas('.\Dataset\\' + path_l + '_Test_NoLabels.xml')
    df_no_label, _ = data_pre_treatment(df_no_label)

    # Load data with label
    df_Label = get_datas('.\Dataset\\' + path_l + '_Test_Gold.xml')
    _, label_gold = data_pre_treatment(df_Label)

    # Load the model
    saved_model = tf.keras.models.load_model(path_l + '_model.h5')

    # On fait la prediction sur les données du fichier Test_NoLabels
    l_predictions = make_prediction_noLabel(df_no_label, saved_model)

    # Evaluate the model
    l_metrics = saved_model.evaluate(l_predictions, label_gold[0])
    print("""
    //////////////////////////////////////////////////
    //////////// PREDICTION LAPTOP ///////////////////
    //////////////////////////////////////////////////
    """)
    print(str(saved_model.metrics_names[1]) + ': ' + str(l_metrics[1]))
