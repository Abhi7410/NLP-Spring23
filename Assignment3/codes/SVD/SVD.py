from sklearn.decomposition import TruncatedSVD
import numpy as np
from tqdm import tqdm
import json
from collections import Counter
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

import pickle

with open("../../Data/smallData.pkl", "rb") as fp:
    smallData = pickle.load(fp)


#======= Co-occurrence Matrix =========

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(smallData)
features_names = vectorizer.get_feature_names_out()

## saving the features list 
with open("../../Data/features_names_small.txt", "w") as fp:
    fp.write("\n".join(features_names))

# create co-occurence matrix
co_occurence_matrix = X.T.dot(X)
# set the diagonal to zero
co_occurence_matrix.setdiag(0)

# saving the co-occurence matrix
sparse.save_npz("../../Data/co_occurenceMatrix_small.npz", co_occurence_matrix)

co_occurence_matrix = sparse.load_npz(
    "../../Data/co_occurenceMatrix_small.npz")

Svd = TruncatedSVD(n_components=300, n_iter=5, random_state=42)
svd_matrix = Svd.fit_transform(co_occurence_matrix)
print("SVD matrix shape: ", svd_matrix.shape)
# print("Explained variance: ", Svd.explained_variance_ratio_)
print("Sum of explained variance: ", np.sum(Svd.explained_variance_ratio_))
print("Eigen values: ", Svd.singular_values_)
