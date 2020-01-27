import fasttext
import numpy as np


def predict(model_path, word_list, k=1005):

    model = fasttext.load_model(model_path)
    result = model.predict(word_list, k)

    score = np.zeros(917)

    for j in range(len(word_list)):
        for i in range(min([k, len(result[0][j])])):
            id = int(result[0][j][i].replace("__label__" , "").replace("," , ""))
            score[id] += result[1][j][i]

    return score
