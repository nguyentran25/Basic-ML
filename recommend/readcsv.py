import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

r_cols = ['books_id', 'users_id', 'rating']
ratings_base = pd.read_csv('/home/nguyentran25/PycharmProjects/recommend/Data/ratings.csv', sep=',', names=r_cols, encoding='latin-1')
# print ratings_test
ratings_base = ratings_base.as_matrix()
print ratings_base
rate1, rate2 = train_test_split(
     ratings_base, test_size=200000)

print X_test.shape[0]
# ratings_base['split'] = np.random.randn(ratings_base.shape[0],1)
# msk = np.random.rand(len(ratings_base))<=0.7
# train = ratings_base[msk]
# test = ratings_base[~msk]

# print test
