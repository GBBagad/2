# Install required libraries
pip install gensim numpy pandas

# Import necessary packages
import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load data
with open('document3.txt', 'r', encoding='utf-8') as file:
    data = file.read()

print(data)

# Preprocess data
stop_words = set(ENGLISH_STOP_WORDS)

def preprocess_doc(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return words

# Convert document into sentences
sentences = [preprocess_doc(data)]

# Define CBOW Word2Vec model
word2vec = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    sg=0
)

# Get most frequent word
word = word2vec.wv.index_to_key[0]
print(word)

# Display word embedding
print("Word Embedding:")
print(word2vec.wv[word])

# Display similar words
print("Similar words:")
print(word2vec.wv.most_similar(word, topn=10))