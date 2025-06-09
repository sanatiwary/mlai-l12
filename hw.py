import pandas as pd
from sklearn.model_selection import train_test_split

movieData = pd.read_csv("movies_metadata.csv")
movieData = movieData[["title", "overview"]]
movieData = movieData.dropna()

x = movieData["overview"]
y = movieData["title"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=1)

import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

wnl = WordNetLemmatizer()

def transform(data):
    corpus = []

    for i in data:
        newi = re.sub("[^a-zA-Z]", " ", i)
        newi = newi.lower()
        newi = newi.split()
        list1 = [wnl.lemmatize(word) for word in newi if word not in stopwords.words("english")]
        corpus.append(" ".join(list1))

    return corpus

xTrainCorpus = transform(xTrain)
xTestCorpus = transform(xTest)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(1, 2))
xTrainNew = cv.fit_transform(xTrainCorpus)
xTestNew = cv.transform(xTestCorpus)

from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5, metric="cosine")
nn.fit(xTrainNew)

def recommendMovie(text):
    trText = transform(text)
    trText = cv.transform(trText)
    dists, indices = nn.kneighbors(trText)
    for i in indices[0]:
        print(yTrain.iloc[i])

str1 = ["a cowboy doll is threatened by a new spaceman toy"]
str2 = ["kids find a magical board game that unleashes wild animals"]

recommendMovie(str1)
recommendMovie(str2)
