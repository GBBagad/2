# Install required libraries
pip install tensorflow gensim pandas numpy scikit-learn

# Import necessary packages
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load dataset
data = pd.read_csv('IMDB Dataset.csv')
print(data.shape)
print(data.head())

# Preprocess sentiment labels
data['sentiment'] = data['sentiment'].str.lower().map({
    'positive': 1,
    'negative': 0
})

# Preprocess reviews
stop_words = set(ENGLISH_STOP_WORDS)

def preprocess_review(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return words

sentences = [preprocess_review(review) for review in data['review']]

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    sentences,
    data['sentiment'],
    test_size=0.2,
    random_state=42
)

# Define CBOW Word2Vec model
word2vec = Word2Vec(
    sentences=X_train,
    vector_size=100,
    window=5,
    min_count=2,
    sg=0
)

# Convert documents into embeddings
def doc_vec(sentence):
    vectors = [
        word2vec.wv[word] for word in sentence if word in word2vec.wv
    ]

    if len(vectors) == 0:
        return np.zeros(100)

    return np.mean(vectors, axis=0)

# Generate document vectors
X_train = np.array([doc_vec(s) for s in X_train])
X_test = np.array([doc_vec(s) for s in X_test])

# Convert labels into NumPy arrays
y_train = y_train.values
y_test = y_test.values

# Define FNN architecture
model = keras.Sequential([
    keras.Input(shape=(100,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)