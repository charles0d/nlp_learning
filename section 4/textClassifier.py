# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:12:10 2022

@author: E030751
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

with open('edgar_allan_poe.txt') as f:
    poe = [x for x in f.read().lower().split('\n') if x != '']

with open('robert_frost.txt') as f:
    frost = [x for x in f.read().lower().split('\n') if x != '']

poeTokens = [[i]+x.split(' ') for (i,x) in enumerate(poe)]
frostTokens = [[i]+x.split(' ') for (i,x) in enumerate(frost)]

poe_train, poe_test = train_test_split(poeTokens)
frost_train, frost_test = train_test_split(frostTokens)

mapping = {}

c=1

texts_train = [poe_train, frost_train]
for text in texts_train:
    for line in text:
        for word in line[1:]:
            if word not in mapping:
                mapping[word] = c
                c += 1

convertToIndex = lambda text: [mapping[x] if x in mapping else 0 for x in text]
poe_train_index = [convertToIndex(line[1:]) for line in poe_train]
poe_test_index = [convertToIndex(line[1:]) for line in poe_test]
frost_train_index = [convertToIndex(line[1:]) for line in frost_train]
frost_test_index = [convertToIndex(line[1:]) for line in frost_test]


def computeTransitions(train_index):
    M = 1+len(mapping)
    N = len(train_index)
    transitions = np.ones((M, M))
    p0 = np.ones((M, 1)) / (M+N)
    counts = np.zeros((M, 1))
    
    for line in train_index:
        if len(line) == 1:
            continue
        p0[line[1]] += 1
        for (i, token) in enumerate(line[1:]):
            counts[token] += 1
            if len(line) > i+2:
                transitions[token, line[i+2]] += 1
    
    for i in range(M):
        token = i
        transitions[i,:] = np.log(transitions[i,:] / (counts[i] + M))
        
    return transitions, np.log(p0)

Apoe, p0poe = computeTransitions(poe_train_index)
Afrost, p0frost = computeTransitions(frost_train_index)
pifrost = np.log(len(frost_train) / (len(frost_train) + len(poe_train)))
pipoe = np.log(1 - pifrost)



def posterior(line, author):
    if len(line) == 1:
        return -1
    if author=='Poe':
        return p0poe[line[1]] + np.sum([Apoe[line[i], line[i+1]] for i in range(len(line)-2)])
    else:
        return p0frost[line[1]] + np.sum([Afrost[line[i], line[i+1]] for i in range(len(line)-2)])
    
    
def predict(line):
    poeProb = posterior(line, 'Poe')
    poeFrost = posterior(line, 'Frost')
    if poeProb > poeFrost:
        return 'Poe'
    else:
        return 'Frost'
    
# Use F1 metrics to compute accuracy of this prediction on test dataset.
# Probably improvable a lot using tokenization