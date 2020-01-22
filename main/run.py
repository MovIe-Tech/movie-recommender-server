import numpy as np
import pandas as pd
from . import lsi
from . import doc2vec
from .preprocess import preprocess_TextToList


def search_for_movies(query, topn=10, w_r_d=1, w_r_f=1, w_r_l=1, w_r_t=1):
    title_list = pd.read_csv("data/movies_data.csv")['title'].values.tolist()
    rate_list = pd.read_csv("data/movies_data.csv")['rate'].values.tolist()
    query_list = preprocess_TextToList(query)
    review_pred = doc2vec.predict_movies(input_list = query_list, topn=1005, model_path='data/doc2vec_reviews.model')
    pred_list = review_pred
    sorted_id = np.argsort(pred_list)[::-1]
    return np.array(title_list)[np.array(sorted_id)][:topn], np.array(rate_list)[np.array(sorted_id)][:topn]