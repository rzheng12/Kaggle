import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
train=pd.read_csv("../input/train.csv")

from sklearn.model_selection import train_test_split
train,dev=train_test_split(train,test_size=0.05,random_state=42)

def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]                                   
    X_indices = np.zeros((m, max_len))
    for i in range(m):                              
        sentence_words = [w.lower() for w in X[i,0].split()]
        j = 0        
        for w in sentence_words:
            w=w.replace(',',"").replace("?","").replace("!","").replace(".","").replace('"',"").replace("'","").replace("’","").replace("(","").replace(")","")
            if w not in word_to_index:
                continue
            X_indices[i, j] = word_to_index[w]
            j += 1
    return X_indices



def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = ''.join(line[:-300])
            words.add(curr_word)
            coefs = np.asarray(line[-300:], dtype='float32')
            word_to_vec_map[curr_word] = coefs
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

word_to_index, _, word_to_vec_map = read_glove_vecs('../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')




train_indices=sentences_to_indices(train.question_text.values.reshape(-1,1),word_to_index,60)
dev_indices=sentences_to_indices(dev.question_text.values.reshape(-1,1),word_to_index, 60)


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        if word=="":
            emb_matrix[index, :] = np.hstack((word_to_vec_map[""],np.zeros(298)))
        else:
            emb_matrix[index, :] = word_to_vec_map[word]
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer



def net(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(input_shape, dtype='int32')
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)   
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(2)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    return model

import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
model = net((60,), word_to_vec_map, word_to_index)
model.summary()



train_Y=keras.utils.to_categorical(train.target.values,2)
dev_Y=keras.utils.to_categorical(dev.target.values,2)
weight_zero=np.sum(train.target.values)/train.shape[0]
train=None
weight={0:weight_zero,1:1}
print(weight)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(train_indices, train_Y, epochs = 50,validation_data=(dev_indices,dev_Y), class_weight=weight,batch_size = 128, shuffle=True)
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
word_to_index=None
word_to_vec_map=None

