!pip install tensorflow gensim pandas numpy

import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

data = pd.read_csv('twitter_dataset.csv')

print(data.shape)
print(data.head())

stop_words = set(ENGLISH_STOP_WORDS)

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return words

sentences = [preprocess_text(tweet) for tweet in data['Text']]

word2vec = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    sg=0
)

if 'good' in word2vec.wv:
    print('Word Embeddings:')
    print(word2vec.wv['good'])

    print('Similar Words:')
    print(word2vec.wv.most_similar('good', topn=5))
else:
    print("'good' is not present in the vocabulary")