import json
import nltk
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import pickle
from nltk.stem import WordNetLemmatizer
# 1. Create the correct absolute path to the 'nltk_data' folder
nltk_path = os.path.join(os.path.dirname(__file__), "nltk_data")

# 2. Tell NLTK to look for resources in that folder
nltk.data.path.append(nltk_path)

nltk.download('punkt')
nltk.download('wordnet')

with open('data/intents.json') as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()

corpus = []
tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
        corpus.append(" ".join(tokens))
        tags.append(intent['tag'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

encoder = LabelEncoder()
y = encoder.fit_transform(tags)

model = MultinomialNB()
model.fit(X, y)

# Save model and encoders
os.makedirs("model", exist_ok=True)
model = pickle.load(open("model/classifier.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
encoder = pickle.load(open("model/label_encoder.pkl", "rb"))


print("Model trained and saved.")
