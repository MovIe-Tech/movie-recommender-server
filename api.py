from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from gensim.models import word2vec
import json
import MeCab
from main.run import search_for_movies

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

@app.route('/reply', methods=['GET'])
def reply():
    text = request.args.get('input')
    (titles, rates) = search_for_movies(text, topn=5)
    return jsonify({
        "1": titles[0],
        "2": titles[1],
        "3": titles[2],
        "4": titles[3],
        "5": titles[4],
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=8888)
