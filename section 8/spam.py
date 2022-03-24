# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:30:08 2022

@author: E030751
"""

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df.columns = ['string_labels', 'data']

df['labels'] = df['string_labels'].map({'ham': 0, 'spam': 1})
df = df.drop('string_labels', axis=1)

X_train0, X_test0, y_train, y_test = train_test_split(df['data'], 
                                                    df['labels'], 
                                                    test_size =0.3)
model = MultinomialNB()
#vectorizer = TfidfVectorizer(decode_error='ignore')
vectorizer = CountVectorizer(decode_error='ignore')

X_train = vectorizer.fit_transform(X_train0)
X_test = vectorizer.transform(X_test0)

model.fit(X_train, y_train)
Ptrain = model.predict(X_train)
Ptest = model.predict(X_test)
print(f'TRAIN f1 score: {f1_score(y_train, Ptrain)}')
print(f'TEST f1 score: {f1_score(y_test, Ptest)}')

Probs_train = model.predict_proba(X_train)[:, 1]
Probs_test = model.predict_proba(X_test)[:, 1]
print(f'TRAIN ROC AUC score: {roc_auc_score(y_train, Ptrain)}')
print(f'TEST ROC AUC score: {roc_auc_score(y_test, Ptest)}')



X = vectorizer.transform(df['data'])
df['predictions'] = model.predict(X)

undetected = df[(df['predictions'] == 0) & (df['labels'] == 1)]




def visualize(label):
  words = ''
  for msg in df[df['labels'] == label]['data']:
    msg = msg.lower()
    words += msg + ' '
  wordcloud = WordCloud(width=600, height=400).generate(words)
  plt.imshow(wordcloud)
  plt.axis('off')
  plt.show()

visualize(0)
visualize(1)