from spacy_model.en_core_web_sm import en_core_web_sm


def get_named_entity(text):
    nlp = en_core_web_sm.load()
    doc = nlp(text)

    named_entity = []

    for ent in doc.ents:
        # print(ent.text, ent.label_)
        ne = ent.text
        named_entity.append(ne)

    return named_entity


text = 'The United States'

print(get_named_entity(text))

