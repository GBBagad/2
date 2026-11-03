# Install required libraries
pip install gensim pandas numpy scikit-learn

# Import necessary packages
import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# Prepare dataset
document = [
    "Machine Learning is a branch of Artificial Intelligence",
    "Cricket and Football are popular sports",
    "Deep Learning uses Neural Networks",
    "Artificial Intelligence helps machines learn",
    "Neural Networks are used in Deep Learning"
]

# Preprocess text
stop_words = set(ENGLISH_STOP_WORDS)

def preprocess_doc(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return words

# Convert documents into sentences
sentences = [preprocess_doc(s) for s in document]
print(sentences)

# Define CBOW Word2Vec model
word2vec = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=0
)

# Generate document embeddings
def doc_vec(sentence):
    vectors = [
        word2vec.wv[word] for word in sentence if word in word2vec.wv
    ]

    if len(vectors) == 0:
        return np.zeros(100)

    return np.mean(vectors, axis=0)

# Convert documents into embeddings
doc_embeddings = np.array([doc_vec(s) for s in sentences])

# Take user input
user_input = input("Enter your search: ")

# Preprocess user input
user_words = preprocess_doc(user_input)

# Convert user input into embedding
user_embeddings = [doc_vec(user_words)]

# Calculate cosine similarity
similarities = cosine_similarity(
    user_embeddings,
    doc_embeddings
)[0]

# Get descending order of similarities
desc_order = sorted(
    range(len(similarities)),
    key=lambda i: similarities[i],
    reverse=True
)

# Recommend top 3 documents
print("Similar Documents:")

for i in desc_order[:3]:
    print("Document:", document[i])
    print("Similarity:", similarities[i])