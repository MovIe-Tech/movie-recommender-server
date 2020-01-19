import pandas as pd
import numpy as np
import MeCab
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from gensim.similarities.docsim import MatrixSimilarity

# LSIにより類似度のnumpyリストを出力
def predict_movies(input_list, corpus_path='data/corpus.txt',
                   dic_path='data/dic.dict'):
    f = open(corpus_path,"rb")
    corpus = pickle.load(f)
    dic = Dictionary.load(dic_path)
    dic.add_documents([input_list])
    corpus.append(dic.doc2bow(input_list))
    lsi = LsiModel(corpus,num_topics=200, id2word=dic)
    vectorized_corpus = lsi[corpus]
    doc_index = MatrixSimilarity(vectorized_corpus)
    sims = doc_index[vectorized_corpus]
    return sims[-1:][0][:-1]