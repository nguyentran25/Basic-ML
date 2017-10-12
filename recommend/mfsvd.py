import pandas as pd 
import numpy as np
from numpy import linalg as LA
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 
from sklearn.model_selection import train_test_split

class SVD_RS(object):
    """docstring for CF"""
    def __init__(self, Y_data, K, user_based = 1):
        self.Y_data = Y_data
        self.K = K
        self.user_based = user_based
        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(Y_data[:, 0])) + 1 
        self.n_items = int(np.max(Y_data[:, 1])) + 1            
        #self.all_users = self.Y_data[:,0] # all users (may be duplicated)
        self.n_ratings = Y_data.shape[0]
        # normalized data
        self.Ybar_data = self.Y_data.copy().astype(np.float32)

    def normalize_Y(self):
        if self.user_based:
            user_col = 0
            item_col = 1
            n_objects = self.n_users

        # if we want to normalize based on item, just switch first two columns of data
        else: # item bas
            user_col = 1
            item_col = 0 
            n_objects = self.n_items

        users = self.Y_data[:, user_col] 
        self.mu = np.zeros((n_objects,))
        for n in xrange(n_objects):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data[ids, 1] 
            # and the corresponding ratings 
            ratings = self.Y_data[ids, 2].astype(np.float32)
            # take mean
            m = np.mean(ratings) 
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            # normalize
            self.Ybar_data[ids, 2] = ratings - self.mu[n]
#             print self.Ybar_data

        ################################################
        # form the rating matrix as a sparse matrix. Sparsity is important 
        # for both memory and computing efficiency. For example, if #user = 1M, 
        # #item = 100k, then shape of the rating matrix would be (100k, 1M), 
        # you may not have enough memory to store this. Then, instead, we store 
        # nonzeros only, and, of course, their locations.
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.todense()


    
    def fit(self): 
        """
        matrix factorization using SVD
        """
        self.normalize_Y()
        U, S, V = LA.svd(self.Ybar)
        Uk = U[:, :self.K]
        Sk = S[:self.K]
        Vkt = V[:self.K, :]
        self.res = Uk.dot(np.diag(Sk)).dot(Vkt)
        
    def pred(self, u, i):
        """ 
        predict the rating of user u for item i 
        if you need the un
        """
        u = int(u)
        i = int(i)
        
        if self.user_based:
            bias = self.mu[u]
        else: 
            bias = self.mu[i]
        pred = self.res[i, u] + bias 
        if pred < 1:
            return 1 
        if pred > 5: 
            return 5 
        return pred 
        
    
    def pred_for_user(self, user_id):
        ids = np.where(self.Y_data_n[:, 0] == user_id)[0]
        items_rated_by_u = self.Y_data_n[ids, 1].tolist()              
        
        y_pred = self.X.dot(self.W[:, user_id]) + self.mu[user_id]
        predicted_ratings= []
        for i in xrange(self.n_items):
            if i not in items_rated_by_u:
                predicted_ratings.append((i, y_pred[i]))
        
        return predicted_ratings
    
    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0 # squared error
        for n in xrange(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2])**2 

        RMSE = np.sqrt(SE/n_tests)
        return RMSE

# Data Movies
# r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
# ratings_base = pd.read_csv('./Data/ub.base', sep='\t', names=r_cols, encoding='latin-1')
# ratings_test = pd.read_csv('./Data/ub.test', sep='\t', names=r_cols, encoding='latin-1')
# rate_train = ratings_base.as_matrix()
# rate_test = ratings_test.as_matrix()

# Data books
r_cols = ['user_id', 'books_id', 'rating']
ratings_base = pd.read_csv('./Data/ratings.csv', sep = ',', names = r_cols, encoding = 'latin-1')
ratings_base = ratings_base.as_matrix()
ratings_base = ratings_base[:, [1, 0, 2]]
ratings_base, tmp = train_test_split(ratings_base, test_size = 0.9)
rate_train, rate_test = train_test_split(ratings_base, test_size = 0.2)

# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1       
print rate_test.shape[0]
print rate_train.shape[0]


rs = SVD_RS(rate_train, K = 10, user_based = 1)
rs.fit()
# evaluate on test data
RMSE = rs.evaluate_RMSE(rate_test)
print '\nUser-based MF, RMSE =', RMSE

# rs = SVD_RS(rate_train, K = 10, user_based = 0)
# rs.fit()
# # evaluate on test data
# RMSE = rs.evaluate_RMSE(rate_test)
# print '\nItem-based MF, RMSE =', RMSE