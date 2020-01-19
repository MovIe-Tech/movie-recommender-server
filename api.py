from flask import Flask, jsonify, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from gensim.models import word2vec
import json
import MeCab
from main import lsi
from main import preprocess

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/reply', methods=['GET'])
def reply():
    text = request.args.get('input')
    input_list = preprocess.preprocess_TextToList(text)
    pred_list = lsi.predict_movies(input_list)
    titles_list = pd.read_csv("data/movie_titles.csv").values.tolist()
    sorted_id = np.argsort(pred_list)[::-1]
    return jsonify({
        "1": titles_list[sorted_id[0]],
        "2": titles_list[sorted_id[1]],
        "3": titles_list[sorted_id[2]],
        "4": titles_list[sorted_id[3]],
        "5": titles_list[sorted_id[4]],
    })


@app.route('/analysis', methods=['GET'])
def analysis():
    text = request.args.get('input')
    input_list = preprocess.analysis(text)
    return jsonify(input_list)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=8888)
