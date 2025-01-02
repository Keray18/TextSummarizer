from flask import Flask, request, jsonify, render_template

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE

import re
import pickle
import string
import os



app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

print(current_dir)

model_path = os.path.join(current_dir, "Notebook", "model.pkl")
vectorizer_path = os.path.join(current_dir, "Notebook", "tfidf.pkl")
scaler_path = os.path.join(current_dir, "Notebook", "scaler.pkl")

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
    
with open(vectorizer_path, 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)
    
with open(scaler_path, 'rb') as scaler_file:
    scaled = pickle.load(scaler_file)

def preprocess_text(text):

    lower = text.lower()
    tokens = word_tokenize(lower)

    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if not re.match(r'\d+', word)]

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    lemma = WordNetLemmatizer()
    lemma_tokens = [lemma.lemmatize(word) for word in filtered_tokens]

    return ' '.join(lemma_tokens)


def get_summary(text):
  sentences = [preprocess_text(sent) for sent in sent_tokenize(text)]
    
  sent_vector = tfidf.transform(sentences).toarray()
  sent_vector = scaled.transform(sent_vector)
  predictions = model.predict(sent_vector)
  relevant_sentences = [sent for sent, preds in zip(sentences, predictions) if preds == 1]
  summary = " ".join(relevant_sentences)

  return summary


@app.route('/prediction', methods=['POST'])
def prediction():
    data = request.get_json()
    sentence = data.get('sentence')
    print(f"Type of text: {type(sentence)}, Value: {sentence}")

    print(f"Sentence: {sentence}")

    summary = get_summary(sentence)
    # print(f"Summary: {summary}")
    
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True, port=5000)