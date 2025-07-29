from flask import Flask, render_template, request, jsonify
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
import random  # <-- Add this at the top
nltk.data.path.append("nltk_data")

app = Flask(__name__)

# Load models
model = pickle.load(open("model/classifier.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
encoder = pickle.load(open("model/label_encoder.pkl", "rb"))

# Load intents
with open("data/intents.json") as f:
    intents = json.load(f)

lemmatizer = WordNetLemmatizer()

def predict_intent(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    X = vectorizer.transform([" ".join(tokens)])
    tag = encoder.inverse_transform(model.predict(X))[0]

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return tag, intent['responses']
    return "unknown", ["Sorry, I didn’t understand that."]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    userText = request.form["msg"]
    tag, responses = predict_intent(userText)
    return random.choice(responses)  # return a random response
import os



if __name__ == "__main__":
   port = int(os.environ.get("PORT", 10000))
   app.run(host='0.0.0.0', port=port)
