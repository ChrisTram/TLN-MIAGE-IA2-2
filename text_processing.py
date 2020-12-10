
from spacy_model.en_core_web_sm import en_core_web_sm
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def get_named_entity(text):
    nlp = en_core_web_sm.load()
    doc = nlp(text)

    named_entity = []

    for ent in doc.ents:
        # print(ent.text, ent.label_)
        ne = ent.text
        named_entity.append(ne)

    return named_entity


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def get_sentiment(word, tag):
    """ returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. """

    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []

    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score()]

def clean_string(str) :
    str = str.replace(',', ' ')
    str = str.replace('.', ' ')
    str = str.replace("!", '')
    str = str.replace("?", '')
    str = str.replace("-", '')
    str = str.replace('(', '')
    str = str.replace("'s", '')
    str = str.replace("'", '')
    str = str.replace(";", ' ')
    str = str.replace("*", '')
    str = str.replace('"', '')
    str = str.replace(":", '')
    str = str.replace(')', '')
    str = str.replace('/', ' ')
    return str

def get_score(sentence, index):

    sentence[:] = [x for x in sentence if x]
    pos_vals = pos_tag(sentence)
    #print(pos_vals)
    senti_vals = [get_sentiment(x, y) for (x, y) in pos_vals]
    #print(senti_vals)

    senti_vals_f = []
    if index > 2 :
        i = index-3
    else :
        i = 0

    while i < index+4 and i < len(pos_vals):
        senti_vals_f.append(senti_vals[i])
        i += 1
    return senti_vals_f

def get_sentiment_score(term, sentence):
    sentence = clean_string(sentence)
    words = sentence.split()
    if len(term.split()) > 1:
        term = term.split()[0]
    term = clean_string(term)
    index = words.index(term)

    scores = get_score(sentence.split(" "),index)

    pos = 0
    neg = 0
    #print(scores)

    for score in scores:
        #print(score)
        if len(score) > 1 :
            if score[0] > 0.5 :
                pos += 1
            if score[1] > 0.5 :
                neg += 1

    if pos > neg :
        return 'positive'
    elif neg > pos :
        return 'negative'
    else :
        return 'neutral'
