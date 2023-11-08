import nltk
nltk.download('stopwords')
nltk.download('punkt')
import jinja2
import numpy as np
import json
import tensorflow
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
import re
import unidecode
import contractions 
from string import punctuation

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from flask import Flask, render_template, request, jsonify, url_for

import inspect

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

from invoke import task

app = Flask(__name__)

# POST method for form result and GET for direct URL access and URL parameters
@app.route('/button_clicked', methods=['POST' , 'GET'])
def button_clicked():
    return render_template('index_test.html')

@app.route('/', methods=['POST' , 'GET'])
def home():
    return render_template('home_google.html')

@app.route('/', methods=['POST','GET'])
def index():
    return render_template('home_google.html')

@app.route('/symptoms', methods=["POST", "GET"])
def symptoms():
    return render_template('Depression_Symptoms.html')

@app.route('/remedies', methods=["POST", "GET"])
def remedies():
    return render_template('Depression_Remedies.html')

@app.route('/redirect', methods=["POST", "GET"])
def redirect():
    p = request.form.get('polarity')
    return render_template('Depression_Remedies.html', polarity=float(p))


@app.route('/predict', methods=['POST'])
def predict(text=None):
    # Get the input data from the request
    text = request.form['speech']
    # Pass the input to the machine learning model
    text_preprocessed = preprocess(text)
    test_text = padding(text_preprocessed)
    prediction, polarity = testing(test_text)
    return render_template('index_test.html', prediction=prediction, text=text, polarity=polarity)


def padding(txt):
    with open('tokenizer_config.json', 'r') as f:
        loaded_tokenizer_config = json.load(f)
    max_seq_length = 100
    loaded_tokenizer = Tokenizer(num_words=5000)
    loaded_tokenizer.word_index = loaded_tokenizer_config['word_index']

    loaded_X_train_sequences = loaded_tokenizer.texts_to_sequences([txt])

    loaded_X_train_padded = pad_sequences(loaded_X_train_sequences, maxlen=max_seq_length, padding='post', truncating='post')
    return loaded_X_train_padded

def preprocess(text):


    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    html_pattern = r'<.*?>'
    text = re.sub(pattern=html_pattern, repl=' ', string=text)

    url_pattern = r'https?://\S+|www\.\S+'
    text = re.sub(pattern=url_pattern, repl=' ', string=text)

    # numbers
    number_pattern = r'\d+'
    text = re.sub(pattern=number_pattern, repl=' ', string=text)

    # unidecode
    text = unidecode.unidecode(text)

    # Expanding Contractions
    text = contractions.fix(text)

    # remove punctutation
    text = text.translate(str.maketrans('', '', punctuation))

    # removing single characters
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    text = re.sub(pattern=single_char_pattern, repl=" ", string=text)

    # Extra spaces
    space_pattern = r'\s+'
    text = re.sub(pattern=space_pattern, repl=" ", string=text)

    # Tokenize and remove stopwords
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]

    # Join the words back into a single string
    text = ' '.join(words)

    return text

model = tensorflow.keras.models.load_model('my_model.h5')

def testing(txt):
    ans = model.predict(txt)
    if ans < 0.5:
        return("Depressed", ans)
    else:
        return("Not Depressed", ans)

if __name__ == '__main__':
    app.run(debug=True)
