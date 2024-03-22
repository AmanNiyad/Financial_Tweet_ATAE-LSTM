import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
# from keras_self_attention import SeqSelfAttention
import math
import nltk
from nltk.corpus import stopwords
import tensorflow as tf

df = pd.read_csv("sent_train.csv")
stop_words = set(stopwords.words('english'))

df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
handle_regex= "^@?(\w){1,15}$"


df['text']=df['text'].str.replace(url_regex, '')
df['text']=df['text'].str.replace(handle_regex, '')

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    st = ""
    for w in w_tokenizer.tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st
df['text'] = df.text.apply(lemmatize_text)

reviews = df['text'].values
labels = df['label'].values
encoder = LabelEncoder()

encoded_labels = encoder.fit_transform(labels)

train_sentences, test_sentences, train_labels, test_labels = train_test_split(reviews, encoded_labels, test_size=0.2, stratify = encoded_labels)

vocab_size = 5068 
oov_tok = ""
max_len = 300
embedding_dim = 100
padding_type = 'post'
trunc_type = 'post'
units = 64
hidden_dim = 64

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding='post', maxlen=max_len)

def dot_product_attention(h, k, v):
    scores = tf.matmul(h, k, transpose_b=True) / np.sqrt(h.shape[-1])  # Scaled dot-product
    attention_weights = tf.nn.softmax(scores, axis=-1)  # Apply softmax for weights
    return attention_weights

#
# model = keras.Sequential([
#     keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
#     keras.layers.Bidirectional(keras.layers.LSTM(64)),
#     keras.layers.Dense(1, activation='tanh'),
#     keras.layers.Flatten(),
#     keras.layers.Activation(activation='softmax'),
#     keras.layers.RepeatVector(units*2),
#     keras.layers.Permute([2,1]),
#     keras.layers.Dropout(0.2),
#     # keras.layers.Dense(3, activation='sigmoid'),
# ])

inputs = keras.Input(shape=300,)
embedded = keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len)(inputs)
lstm_output =  keras.layers.Bidirectional(keras.layers.LSTM(64))(embedded)

query = tf.keras.layers.Dense(units=hidden_dim)(lstm_output)
key = tf.keras.layers.Dense(units=hidden_dim)(lstm_output)

attention_weights = dot_product_attention(query, key, lstm_output)
attention_weights = tf.expand_dims(attention_weights, axis=-1)
context_vector = attention_weights * lstm_output
context_vector = tf.reduce_sum(context_vector, axis=1)

model = keras.Model(inputs=inputs, outputs=context_vector)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model.summary()
num_epochs = 5
history = model.fit(train_padded, train_labels, epochs=num_epochs,verbose=1, validation_split=0.1)
