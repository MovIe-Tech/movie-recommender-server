import pandas as pd
import numpy as np
import MeCab


## 基本形で分かち書きをする関数
def analysis(text):
    mecab = MeCab.Tagger()
    mecab.parse("")
    mecab.parseToNode("dummy")
    node = mecab.parseToNode(text)
    word = ""
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
        pre_feature = node.feature
        node = node.next
    return word[1:]


# ストップワードを用い、テキストを単語のリストにする, 入力に利用
def preprocess_TextToList(text, stopwords_path='data/stop_words.csv'):
    splitted_reviews = analysis(text).split(' ')
    stopwords = pd.read_csv(stopwords_path, encoding='utf-8').T.values.tolist()[0]
    return [word for word in splitted_reviews if word not in stopwords]
