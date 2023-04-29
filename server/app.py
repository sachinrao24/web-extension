import flask
import os
from flask import jsonify, request , render_template
from flask import flash, redirect, url_for, session
from joblib import load
import requests, json
import pandas as pd
import requests
import random
import subprocess
import glob
from random import random
import re
import os
import numpy as np
import sklearn
import joblib
import requests
import time
from torch.utils.data import TensorDataset, DataLoader
from Preprocess import Preprocess as pre

app = flask.Flask(__name__ )
preprocess = Preprocess()

#Classifiers
model = joblib.load('model.pkl')

@app.route("/")
def home():
    return "<h1>Running Flask on Google Colab! with " + 'Hello' + "</h1>" 
  
@app.route('/predict', methods=['POST'])
def predict():
    texts = request.get_json()['data']
    tweet_df = pd.DataFrame(texts)
    tweet_df.columns = ['tweets']

    sentiments = {0:'religion', 1:'age', 2:'ethnicity', 3:'gender', 4:'not_cyberbullying'}

    tweet_texts = []
    for i in range(len(tweet_df)):
        texts = pre.deep_clean(tweet_df['tweets'][i])
        texts = np.array(texts).reshape(-1,1)
        texts = x.flatten()
        texts, text_mask = pre.bert_tokenizer(texts)
        texts = TensorDataset(texts,text_mask)
        texts = DataLoader(texts)
        tweet_texts.append(texts)
    tweet_df['sentiment'] = np.array([pre.bert_predict(model,tweet) for tweet in tweet_texts])
    # 0, 3, 2 1

    sentiment_list = []
    safe_tweets = []

    for sentiment in tweet_df['sentiment']:
        sentiment_list.append(sentiments.get(sentiment))
        if sentiment==4:
            safe_tweets.append(sentiment)

    return jsonify(safe_tweets)


if __name__ == '__main__':
    app.run()