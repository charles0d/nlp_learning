# -*- coding: utf-8 -*-
"""
tf idf test
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

import json

df = pd.read_csv('tmdb_5000_movies.csv')

docs = []

for i, row in df.iterrows():
    doc = row['title'] + ' \n'
    kw = [x['name'] for x in json.loads(row['keywords'])]
    doc += ', '.join(kw) +' \n'
    
    languages = [x['name'] for x in json.loads(row['spoken_languages'])]
    doc += ', '.join(languages) + ' \n'
    
    docs.append(doc)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)


def recommend(name):
    index = df[df['original_title']==name].index[0]
    sims = [cosine_similarity(X[index], x)[0][0] for x in X]
    top6 = np.argsort(sims)[-6:]
    top6_vals = [df.loc[i, 'original_title'] for i in top6]
    return top6_vals
