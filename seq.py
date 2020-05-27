# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:08:44 2019

@author: dell
"""

import numpy as np
from Bio import SeqIO
from nltk import trigrams, bigrams
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import re

from keras import backend as K
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten,Dense,Embedding,Convolution1D,Dropout,Activation,MaxPooling1D
from keras.optimizers import SGD,Adam
from keras.models import load_model
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.utils import plot_model

np.set_printoptions(threshold=np.inf)

def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def specificity(y_true, y_pred):
    
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity


texts = []
for index, record in enumerate(SeqIO.parse('dataset.fasta', 'fasta')):
    tri_tokens = bigrams(record.seq)
    temp_str = ""
    for item in ((tri_tokens)):
        #print(item),
        temp_str = temp_str + " " +item[0] + item[1]
        #temp_str = temp_str + " " +item[0]
    texts.append(temp_str)

seq=[]
stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
for doc in texts:
    doc = re.sub(stop, '', doc)
    seq.append(doc.split())

w2v_model = Word2Vec.load('word2vec.model')
embedding_matrix = w2v_model.wv.vectors

vocab_list = list(w2v_model.wv.vocab.keys())
word_index = {word: index for index, word in enumerate(vocab_list)}

def get_index(sentence):
    global word_index
    sequence = []
    for word in sentence:
        try:
            sequence.append(word_index[word])
        except KeyError:
            pass
    return sequence

X_data = np.array(list(map(get_index, seq)))
Y_data = np.load("y_dataset.npy")



#
maxlen=30

model = Sequential()
model.add(Embedding(
    input_dim=embedding_matrix.shape[0],
    output_dim=embedding_matrix.shape[1],
    input_length=maxlen,
    weights=[embedding_matrix],
    trainable=True))

model.add(Convolution1D(nb_filter=64, filter_length=6, input_shape=(30, 20)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

model.add(Convolution1D(nb_filter=16, filter_length=4))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
#
model.add(Convolution1D(nb_filter=4, filter_length=2))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy',specificity,recall])


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
X_train,X_test, Y_train, Y_test =train_test_split(X_data,Y_data,test_size=0.2, random_state=0)

history=model.fit(X_train, Y_train, epochs=400, batch_size=200,validation_data=(X_test, Y_test),verbose=1,shuffle=True)
model.save('model.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
#plt.savefig("acc3.png")








