from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from gensim.models import LsiModel
from gensim.corpora import Dictionary
import pickle
from gensim.similarities.docsim import MatrixSimilarity
import MeCab

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def index():
    text = request.args.get('input')
    return jsonify(find(text))


def find(text):
    input_list = preprocess_TextToList(text)
    pred_list = predict_movies(input_list)
    titles_list = pd.read_csv("data/movie_titles.csv").values.tolist()
    sorted_id = np.argsort(pred_list)[::-1]
    return {
        '1': titles_list[sorted_id[0]],
        '2': titles_list[sorted_id[1]],
        '3': titles_list[sorted_id[2]],
        '4': titles_list[sorted_id[3]],
        '5': titles_list[sorted_id[4]],
    }


# LSIにより類似度のnumpyリストを出力
def predict_movies(input_list, corpus_path='data/corpus.txt',
                   dic_path='data/dic.dict'):
    f = open(corpus_path, "rb")
    corpus = pickle.load(f)
    dic = Dictionary.load(dic_path)
    dic.add_documents([input_list])
    corpus.append(dic.doc2bow(input_list))
    lsi = LsiModel(corpus, num_topics=200, id2word=dic)
    vectorized_corpus = lsi[corpus]
    doc_index = MatrixSimilarity(vectorized_corpus)
    sims = doc_index[vectorized_corpus]
    return sims[-1:][0][:-1]


## 基本形で分かち書きをする関数
def analysis(text):
    # mecab = MeCab.Tagger("-Ochasen")
    mecab = MeCab.Tagger('-d /virutual/movietech/.local/mecab-ipadic/')
    mecab.parse("")
    mecab.parseToNode("dummy")
    node = mecab.parseToNode(text)
    word = ""
    pre_feature = ""
    while node:
        # 名詞、形容詞、動詞、形容動詞であるかを判定する。
        isUsed = "名詞" in node.feature
        isUsed = "形容詞" in node.feature or isUsed
        isUsed = "動詞" in node.feature or isUsed
        isUsed = "形容動詞" in node.feature or isUsed
        # 以下に該当する場合は除外する。（ストップワード）
        isUsed = (not "代名詞" in node.feature) and isUsed
        isUsed = (not "助動詞" in node.feature) and isUsed
        isUsed = (not "非自立" in node.feature) and isUsed
        isUsed = (not "数" in node.feature) and isUsed
        isUsed = (not "人名" in node.feature) and isUsed
        if isUsed:
            word += " {0}".format(node.feature.split(",")[6])
        '''
        if isUsed:
            if ("名詞接続" in pre_feature and "名詞" in node.feature) or ("接尾" in node.feature):
            word += "{0}".format(node.surface)
        else:
        word += " {0}".format(node.surface)
        #print("{0}{1}".format(node.surface, node.feature))
        '''
        pre_feature = node.feature
        node = node.next
    return word[1:]


# ストップワードを用い、テキストを単語のリストにする, 入力に利用
def preprocess_TextToList(text, stopwords_path='data/stop_words.csv'):
    splitted_reviews = analysis(text).split(' ')
    stopwords = pd.read_csv(stopwords_path, encoding='utf-8').T.values.tolist()[0]
    return [word for word in splitted_reviews if word not in stopwords]


if __name__ == '__main__':
    app.run()
