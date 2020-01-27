from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from gensim.models import word2vec
import json
import MeCab
import random
from main.run import search_for_movies

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)
eval_mode = True

@app.route('/reply', methods=['GET'])
def reply():
    text = request.args.get('input')
    is_randomize = (eval_mode and random.random() > 0.5)
    (titles, rates) = search_for_movies(text, topn=5, randomize=is_randomize)
    return jsonify({
        "1": {
            "title": titles[0],
            "rate" : rates[0], 
        },
        "2": {
            "title": titles[1],
            "rate" : rates[1],
        },
        "3": {
            "title": titles[2],
            "rate" : rates[2],
        },
        "4": {
            "title": titles[3],
            "rate" : rates[3],
        },
        "5": {
            "title": titles[4],
            "rate" : rates[4],
        },
        "randomize": is_randomize
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=8888)
