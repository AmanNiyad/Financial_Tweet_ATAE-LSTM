import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras_self_attention import SeqSelfAttention
import math
import nltk
from nltk.corpus import stopwords

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



pos = 0
neg = 0
for i in range(df.shape[0]):
    if df.iloc[i]['label'] == 1:
        pos = pos + 1
    elif df.iloc[i]['label'] == 0:
        neg = neg + 1
neu = df.shape[0]-(pos+neg)
# print("Percentage of text with positive sentiment is "+str(pos/df.shape[0]*100)+"%")
# print("Percentage of text with negative sentiment is "+str(neg/df.shape[0]*100)+"%")
# print("Percentage of text with neutral sentiment is "+str(neu/df.shape[0]*100)+"%")

#understand using ai

vocab_size = 3000
oov_tok = ""
max_len = 200
embedding_dim = 100

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
def get_word2vec_embedding_matrix(model):
    embedding_matrix = np.zeros((vocab_size,300))
    for word, i in tokenizer.word_index.items():
        try:
            embedding_vector = model[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector
    return embedding_matrix

# print(df)

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
    keras.layers.Bidirectional(keras.layers.LSTM(64,return_sequences=True)),
    keras.layers.LSTM(64,return_sequences=True),
    SeqSelfAttention(attention_activation='softmax'),
    # keras.layers.Flatten(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1000, activation='softmax'),
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
