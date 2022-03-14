import spacy
import pandas as pd
import numpy as np


def get_event_map(nlp, sentence):
    doc = nlp(sentence)
    event_map = {}
    sentence = next(doc.sents)
    for word in sentence:
        if 'subject' not in event_map and word.dep_ in ['nsubj', 'propn', 'pron']:
            event_map['subject'] = word
        elif 'object' not in event_map and word.dep_ in ['dobj']:
            event_map['object'] = word
        elif 'verb' not in event_map and word.dep_ in ['ROOT', 'aux']:
            event_map['verb'] = word
        elif 'wildcard' not in event_map and word.dep_ in ['iobj', 'pobj', 'acomp']:
            event_map['wildcard'] = word
    return event_map


nlp = spacy.load('en_core_web_sm')

df = pd.read_csv("ROC_data.csv")
df.head()

train_split = 20
test_split = 2

train_ratio = int((train_split * 0.01) * len(df))
test_ratio = int((1 - (test_split * 0.01)) * len(df))
train, validate, test = np.split(df.sample(frac=1, random_state=42), [train_ratio, test_ratio])

print("Length of training data", len(train))
print("Length of testing data", len(test))
print("Length of validation data", len(validate))


def get_sentence_file(df, filename):
    print("Total rows for ", filename, ": ", len(df))
    x = 0
    with open(filename, 'w') as w:
        for row in df.itertuples():
            if x % 1000 == 0:
                print("Processing for row: ", x)
            x += 1
            w.write("Story Start\n")
            for i in range(5):
                w.write(row[i + 3] + '\n')
            w.write("Story End\n")


def get_events_file(df, filename):
    print("Total rows for ", filename, ": ", len(df))
    x = 0
    with open(filename, 'w') as w:
        for row in df.itertuples():
            if x % 1000 == 0:
                print("Processing for row: ", x)
            x += 1
            w.write("Story Start\n")
            for i in range(5):
                event_map = get_event_map(nlp, row[i + 3])
                w.write(str(event_map.get('subject', 'NA')) + ' ')
                w.write(str(event_map.get('object', 'NA')) + ' ')
                w.write(str(event_map.get('verb', 'NA')) + ' ')
                w.write(str(event_map.get('wildcard', 'NA')) + ' ')
                w.write('\n')
            w.write("Story End\n")


# Uncomment to generate event files
# get_events_file(train, "train_events.txt")
# get_events_file(test, "test_events.txt")

# Uncomment to generate sentence files
# get_sentence_file(train, "train_sentences.txt")
# get_sentence_file(test, "test_sentences.txt")
