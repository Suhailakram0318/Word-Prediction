from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from textblob import TextBlob
from translate import Translator
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import pyttsx3

app = Flask(__name__)

model = load_model('next_words.h5')
with open('token.pkl', 'rb') as token_file:
    tokenizer = pickle.load(token_file)
translator = Translator(to_lang="te")
engine = pyttsx3.init()

def predict_next_word(text):
    words = text.split()[-3:]
    sequence = tokenizer.texts_to_sequences([words])[0]
    sequence = np.array([sequence])
    preds = np.argmax(model.predict(sequence))
    predicted_word = ""

    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break

    sentiment = TextBlob(predicted_word).sentiment
    sentiment_label = f"Sentiment - Polarity: {sentiment.polarity:.2f}, Subjectivity: {sentiment.subjectivity:.2f}"

    return predicted_word, sentiment_label

def scrape_related_words(query):
    search_url = f"https://www.google.com/search?q={query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, "html.parser")
    search_results = soup.find_all("h3")
    related_words = [result.text for result in search_results]
    return related_words

def get_related_words_tfidf(query, related_words):
    documents = [query] + related_words
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    related_words_with_scores = [(word, score) for word, score in zip(related_words, cosine_similarities)]
    related_words_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in related_words_with_scores]

def get_related_words_sentiments(related_words):
    sentiments = []
    for word in related_words:
        sentiment = TextBlob(word).sentiment
        sentiments.append((word, sentiment.polarity, sentiment.subjectivity))
    return sentiments

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text')
    predicted_word, sentiment_label = predict_next_word(text)
    related_words = scrape_related_words(predicted_word)
    related_words_tfidf = get_related_words_tfidf(predicted_word, related_words)
    related_words_sentiments = get_related_words_sentiments(related_words_tfidf)
    response = {
        'predicted_word': predicted_word,
        'sentiment_label': sentiment_label,
        'related_words': related_words_sentiments
    }
    return jsonify(response)

@app.route('/translate', methods=['POST'])
def translate():
    predicted_word = request.json.get('predicted_word')
    translated_word = translator.translate(predicted_word)
    return jsonify({'translated_word': translated_word})

@app.route('/read_output', methods=['POST'])
def read_output():
    output_text = request.json.get('output_text')
    engine.say(output_text)
    engine.runAndWait()
    return jsonify({'status': 'success'})

if __name__ == "__main__":
    app.run(debug=True)
