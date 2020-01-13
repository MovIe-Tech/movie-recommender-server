import numpy as np
import pandas as pd
from utils.preprocess import preprocess_TextToList
from prediction import lsi


def find(text):
    input_list = preprocess_TextToList(text)
    pred_list = lsi.predict_movies(input_list)
    titles_list = pd.read_csv("data/movie_titles.csv").values.tolist()
    sorted_id = np.argsort(pred_list)[::-1]
    return {
        '1': titles_list[sorted_id[0]],
        '2': titles_list[sorted_id[1]],
        '3': titles_list[sorted_id[2]],
        '4': titles_list[sorted_id[3]],
        '5': titles_list[sorted_id[4]],
    }
