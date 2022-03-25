# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:47:07 2022

@author: E030751
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd

df = pd.read_csv('AirlineTweets.csv').loc[:, ['airline_sentiment', 'text']]

df['labels'] = df['airline_sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})

X_train0, X_test0, y_train, y_test = train_test_split(df['text'], df['labels'], test_size = 0.33)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train0)
X_test = vectorizer.transform(X_test0)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

print('Accuracy')
print(f'train: {accuracy_score(y_train, pred_train)}, test: {accuracy_score(y_test, pred_test)}')
print('f1')
print(f'train: {f1_score(y_train, pred_train, average=None)}, test: {f1_score(y_test, pred_test, average=None)}')

probas_train = model.predict_proba(X_train)
probas_test = model.predict_proba(X_test)

print('roc_auc')
print(f'train: {roc_auc_score(y_train, probas_train, multi_class="ovo")}, test: {roc_auc_score(y_test, probas_test, multi_class="ovo")}')

cm = confusion_matrix(y_test, pred_test, normalize='true')
print(cm)