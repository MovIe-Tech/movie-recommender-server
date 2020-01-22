import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

    
# Doc2Vecにより類似度のnumpyリストを出力
def predict_movies(input_list, topn=1005, model_path='data/doc2vec1005id.model'):
    model = Doc2Vec.load(model_path)
    vec = model.infer_vector(input_list)
    array = np.array(model.docvecs.most_similar([vec], topn=topn))
    return np.ravel(array[array[:,0].argsort(), :][:,1:])