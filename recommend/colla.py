import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_similarity_score
from scipy import sparse
from sklearn.model_selection import train_test_split
import os

class CF(object):
    """docstring for CF"""
    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):
        self.uuCF = uuCF # user-user (1) or item-item (0) CF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k # number of neighbor points
        self.dist_func = dist_func
        self.Ybar_data = None
        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
    
    def add(self, new_data):
        """
        Update Y_data matrix when new ratings come.
        For simplicity, suppose that there is no new user or item.
        """
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)

    def normalize_Y(self):
        users = self.Y_data[:, 0] # all users - first col of the Y_data
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in xrange(self.n_users):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data[ids, 1] 
            # and the corresponding ratings 
            ratings = self.Y_data[ids, 2]
            # take mean
            m = np.mean(ratings) 
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            # normalize
            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        ################################################
        # form the rating matrix as a sparse matrix. Sparsity is important 
        # for both memory and computing efficiency. For example, if #user = 1M, 
        # #item = 100k, then shape of the rating matrix would be (100k, 1M), 
        # you may not have enough memory to store this. Then, instead, we store 
        # nonzeros only, and, of course, their locations.
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()
        return self.Ybar

    def similarity(self):
        # print self.Ybar
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)

    def refresh(self):
        """
        Normalize data and calculate similarity matrix again (after
        some few ratings added)
        """
        print "start"
        self.normalize_Y()
        print "end"
        self.similarity() 
        
    def fit(self):
        self.refresh()

    def __pred(self, u, i, normalized = 1):
        """ 
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        # Step 1: find all users who rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # Step 2: 
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # Step 3: find similarity btw the current user and others 
        # who already rated i
        sim = self.S[u, users_rated_i]
        # Step 4: find the k most similarity users
        a = np.argsort(sim)[-self.k:] 
        # and the corresponding similarity levels
        nearest_s = sim[a]
        # How did each of 'near' users rated item i
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            # add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)

        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
    
    
    def pred(self, u, i, normalized = 1):
        """ 
        predict the rating of user u for item i (normalized)
        if you need the un
        """
        if self.uuCF: return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)

    def recommend(self, u, normalized = 1):
        """
        Determine all items should be recommended for user u. (uuCF =1)
        or all users who might have interest on item u (uuCF = 0)
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which 
        have not been rated by u yet. 
        """
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()              
        recommended_items = []
        for i in xrange(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 4.8: 
                    recommended_items.append(i)
        
        return recommended_items 

    # def print_recommendation(self):
    #     """
    #     print all items which should be recommended for each user 
    #     """
    #     print 'Recommendation: '
    #     for u in xrange(self.n_users):
    #         recommended_items = self.recommend(u)
    #         if self.uuCF:
    #             print '    Recommend item(s):', recommended_items, 'to user', u
    #         else: 
    #             print '    Recommend item', u, 'to user(s) : ', recommended_items
    def print_recommendation(self):
        """
        print all items which should be recommended for each user 
        """
        print 'Recommendation: '
        u = raw_input()
        u = int(u)
        recommended_items = self.recommend(u)
        if self.uuCF:
            print '    Recommend item(s):', recommended_items, 'to user', u
        else: 
            print '    Recommend item', u, 'to user(s) : ', recommended_items

# Data Movies
# r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
# ratings_base = pd.read_csv('./Data/ub.base', sep='\t', names=r_cols, encoding='latin-1')
# ratings_test = pd.read_csv('./Data/ub.test', sep='\t', names=r_cols, encoding='latin-1')
# rate_train = ratings_base.as_matrix()
# rate_test = ratings_test.as_matrix()
# print rate_train


# # Data Books
r_cols = ['books_id', 'users_id', 'rating']
ratings_base = pd.read_csv('./Data/ratings.csv', sep=',', names=r_cols, encoding='latin-1')
ratings_base = ratings_base.as_matrix()
# print ratings_base[0][1]
# ratings_base = ratings_base[:,[1,0,2]]; #swap books_id and users_id
ratings_base = np.array(ratings_base)
print ratings_base
rate_train, rate_test = train_test_split(ratings_base, test_size = 0.2, random_state = None) #split ratings base, rate_train/rate_test = 80/20
# ratings_base, tmp = train_test_split(ratings_base, test_size = 0.9)
# rate_train, rate_test = train_test_split(ratings_base, test_size = 0.1)
# print rate_train
# print rate_test

size = ratings_base.shape[0]
user = 1
train = []
tmp = []
print "Tao data"
for i in xrange(size):
    if ratings_base[i][0] == user:
        if ratings_base[i][2] > 4:
            tmp.append(ratings_base[i][1])

    else:
        user += 1
        train.append(tmp)
        tmp = []
        if ratings_base[i][2] > 4:
            tmp.append(ratings_base[i][1])

train.append(tmp)
print "da tao xong"
print len(train)

# ============================================
# print train[0]
# # indices start from 0
# rate_train[:, :2] -= 1
# rate_test[:, :2] -= 1
# # print rate_train
# #users-users
# rs = CF(rate_train, k = 30, uuCF = 1)
# # rs.fit( )
# print "Chuan hoa du lieu"
# a = rs.normalize_Y()
# print "Da chuan hoa xong"
# # a = a.tolist()
# a = a.T
# # print a.shape[]
# print a
# content = ""
# for i in range (0, 9848):
#     tmp = []
#     for j in range (0,53423):
#         if a[i, j] >=5:
#             tmp.append(j)
#     tmp = str(tmp)
#     content += '[' + '"' + 'U' + str(i+1) + '"' + ',' + ' ' + tmp + ']' + '\n'
#     print "user", i+1, "done"
# with open('input.json', 'w') as f:
#     f.write(content)
# ======================================================

user = 1
content = ""
for i in train:
    content += '[' + '"' + 'U' + str(user) + '"' + ',' + ' ' + str(i) + ']' + '\n'
    print "item", user, "done"
    user += 1
print len(train)
with open('input.json', 'w') as f:
    f.write(content)




# # cnt = 0
# # for i in range(0,943):
# #     print a[i, 0]
# # print a[942,1681]
# # print a[1681, 942]
# print "build similarity table: done"
# n_tests = rate_test.shape[0]
# SE = 0 # squared error
# for n in xrange(n_tests):
#     pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
#     SE += (pred - rate_test[n, 2])**2 

# RMSE = np.sqrt(SE/n_tests)
# print 'User-user CF, RMSE =', RMSE

# while True:
#     rs.print_recommendation()

#items-items
# rs = CF(rate_train, k = 30, uuCF = 0)
# rs.fit()

# n_tests = rate_test.shape[0]
# SE = 0 # squared encodingrror
# for n in xrange(n_tests):
#     pred = rs.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
#     SE += (pred - rate_test[n, 2])**2 

# RMSE = np.sqrt(SE/n_tests)
# print 'Item-item CF, RMSE =', RMSE