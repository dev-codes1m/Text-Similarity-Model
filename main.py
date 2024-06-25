import numpy as np
import pandas as pd
import os
import nltk.corpus
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from flask import Flask,redirect, url_for, request,render_template
import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from flask.json import jsonify
from nltk.stem import WordNetLemmatizer
stop = stopwords.words('english')
stemmer = PorterStemmer()

def text_cleaning(text):
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    text = " ".join([word for word in text.split() if word not in (stop)])
    text = word_tokenize(text)
    stem_words = []
    for word in text:
        stem_words.append(stemmer.stem(word))
        stem_words.append(" ")
    return "".join(stem_words)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return round(((tfidf * tfidf.T).A)[0,1],3)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/textsimilarity',methods = ['POST'])
def text_similarity():
    text1 = request.form['str1']
    text2 = request.form['str2']
    text1 = text_cleaning(text1)
    text2 = text_cleaning(text2)

    similarity = cosine_sim(text1,text2)
    return render_template('home.html',similarity = similarity)

if __name__ =='__main__':
    app.run(port=5000,debug=True)
