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
import math
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
stop_words.add("RT")

stocks=pd.read_csv("stocks_cleaned.csv")
stockerbot=pd.read_csv("stockerbot-export.csv",encoding='utf-8',lineterminator='\n',error_bad_lines=False)
stockerbot=stockerbot.drop(columns=['id'])

stockerbot["timestamp"] = pd.to_datetime(stockerbot["timestamp"])
stockerbot["text"] = stockerbot["text"].astype(str)
stockerbot["url"] = stockerbot["url"].astype(str)

stockerbot["company_names"] = stockerbot["company_names"].astype("category")
stockerbot["symbols"] = stockerbot["symbols"].astype("category")
stockerbot["source"] = stockerbot["source"].astype("category")
stockerbot['date'] = stockerbot['timestamp'].dt.date
stockerbot['time'] = stockerbot['timestamp'].dt.time


stockerbot['text'] = stockerbot['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
stockerbot = stockerbot[stockerbot["source"] != "test5f1798"]

url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
handle_regex= "^@?(\w){1,15}$"


stockerbot['text']=stockerbot['text'].str.replace(url_regex, '')
stockerbot['text']=stockerbot['text'].str.replace(handle_regex, '')

stockerbot= stockerbot[~stockerbot.text.str.contains("BTC")]

print(stockerbot['text'])
