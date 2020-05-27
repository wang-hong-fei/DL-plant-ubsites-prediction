# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:33:37 2020

@author: dell
"""

import numpy as np
from keras.models import load_model

from Bio import SeqIO
from nltk import trigrams, bigrams
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import re


texts = []
for index, record in enumerate(SeqIO.parse('testset.fasta', 'fasta')):
    tri_tokens = bigrams(record.seq)
    temp_str = ""
    for item in ((tri_tokens)):
        #print(item),
        temp_str = temp_str + " " +item[0] + item[1]
        #temp_str = temp_str + " " +item[0]
    texts.append(temp_str)
#
seq=[]
stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
for doc in texts:
    doc = re.sub(stop, '', doc)
    seq.append(doc.split())
#
w2v_model = Word2Vec.load('word2vec.model')
embedding_matrix = w2v_model.wv.vectors
#
vocab_list = list(w2v_model.wv.vocab.keys())
word_index = {word: index for index, word in enumerate(vocab_list)}
#
def get_index(sentence):
    global word_index
    sequence = []
    for word in sentence:
        try:
            sequence.append(word_index[word])
        except KeyError:
            pass
    return sequence
#
X_data = np.array(list(map(get_index, seq)))
#
model = load_model('model.h5')

preds = model.predict_classes(X_data)

for i in preds:
    if i==1:
        print ('ub')
    else:
        print ('non-ub')