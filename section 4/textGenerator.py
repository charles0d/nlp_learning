# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:30:31 2022

@author: E030751
"""

import string
from collections import defaultdict
import random as rand

with open('robert_frost.txt') as f:
    frost = [x.translate(str.maketrans('','',string.punctuation))
                 for x in f.read().rstrip().lower().split('\n') if x != '']
    
corpus = [line.split(' ') for line in frost]
for line in corpus:
    if '' in line:
        line.remove('')

pi = defaultdict(int)
A1 = dict()
A2 = dict()
n = 0
N = len(corpus)

for line in corpus:
    n+=1
    if n % 100 == 0 or n == N:
        print(f'{n} / {N}')
    x0 = line[0]
    pi[x0] += 1
    
    if len(line) == 1:
        continue
    
    x1 = line[1]
    
    if x0 not in A1:
        A1[x0] = defaultdict(int)
    A1[x0][x1] += 1
    
    i=0
    x2 = ''
    while x2 != 'EndOfLine':
        x0 = line[i]
        x1 = line[i+1]
        if len(line) == i+2:
            x2 = 'EndOfLine'
        else:
            x2 = line[i+2]
        
        if x0 not in A2:
            A2[x0] = dict()
        
        if x1 not in A2[x0]:
            A2[x0][x1] = defaultdict(int)
        
        A2[x0][x1][x2] += 1
        
        i+=1
    

# normalize
pi = {k: pi[k]/sum(pi.values()) for k in pi}
for x0 in A1:
    A1[x0] = {k: A1[x0][k]/sum(A1[x0].values()) for k in A1[x0]}

for x0 in A2:
    for x1 in A2[x0]:
        A2[x0][x1] = {k: A2[x0][x1][k]/sum(A2[x0][x1].values()) for k in A2[x0][x1]}
        
        

def sample(d):
    ''' d is a probability dict, return a sample from d'''
    assert(round(sum(d.values()), 8) == 1)
    r = rand.random()
    cum = 0
    for k in d:
        cum += d[k]
        if cum >= r:
            return k
        
    
        
def generate():
    line = []
    line.append(sample(pi))
    line.append(sample(A1[line[0]]))
    i = 0
    while True:
        x2 = sample(A2[line[i]][line[i+1]])
        
        if x2 == 'EndOfLine':
            return line
        line.append(x2)

        i += 1

