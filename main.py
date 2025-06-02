import pandas as pd
import numpy as np

data = pd.read_csv("train.txt")
dataTrain = pd.read_csv("train.txt", delimiter=';', names=["text", "label"])

print(dataTrain.head(10))
print(dataTrain["label"].value_counts())

def customEncoder(data):
    data.replace(to_replace="surprise", value=1, inplace=True)
    data.replace(to_replace="love", value=1, inplace=True)
    data.replace(to_replace="joy", value=1, inplace=True)
    data.replace(to_replace="fear", value=0, inplace=True)
    data.replace(to_replace="anger", value=0, inplace=True)
    data.replace(to_replace="sadness", value=0, inplace=True)

customEncoder(dataTrain["label"])
print(dataTrain.head(10))

import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer

lm = WordNetLemmatizer()

def transformation(data):
    corpus = []

    for sentence in data:
        newItem = re.sub("[^a-zA-Z]", '', str(sentence))
        newItem = newItem.lower()
        newItem = newItem.split()

        newItem = [lm.lemmatize(word) for word in newItem if word not in set(stopwords.words("english"))]

        corpus.append(''.join(str(x) for x in newItem))

    return corpus

corpus = transformation(dataTrain["text"])
print(corpus[1])