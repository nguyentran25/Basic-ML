#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
import data_process
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np

class NewsClassification:
	def __init__(self):
		self.database = data_process.NewsData() #store vietnamese dict and text 's features
		self.clf = MultinomialNB() #naive bayes model
		self.recall_and_precision = OrderedDict() #list of recall and precision score
		self.vietnamese_dict = {}
		self.size_dict = 0

	def input_data(self, path_train, path_test_all):
		print "Nhap 1 de tao tu dien moi, 2 de load tu dien tu file"
		char = raw_input()
		if char == "1":
			self.vietnamese_dict = self.database.create_new_dict(path_train)
			self.vietnamese_dict = self.database.create_new_dict(path_test_all)
			self.database.write_dict_to_file(self.vietnamese_dict)
		if char == "2":
			self.vietnamese_dict = self.database.load_dict_from_file('./dataset/news_dict.txt')
		self.size_dict = self.database.news_dict.get_size_dict()

	def train(self, path_train):
		(train_features, train_label) = self.database.create_matrix_bag_of_words(path_train, self.size_dict, 0)
		self.clf.fit(train_features, train_label)
		return train_features.shape[0]

	def pred(self, test_features):
		return self.clf.predict(test_features)

	def create_true_false_matrix(self, path, true_false_matrix, label):
		#true_false_matrix: matrix n*n, with two dimensions: actual and predicted, n = number of classes
		(test_features, test_label) = self.database.create_matrix_bag_of_words(path, self.size_dict, label)
		y_pred = self.pred(test_features)
		y_pred = np.array(y_pred)
		test_label = np.array(test_label)
		for i in xrange(len(y_pred)):
			if y_pred[i] == label:
				true_false_matrix[label][label] += 1
			else:
				true_false_matrix[y_pred[i]][label] += 1

	def calc_recall_and_precision(self, path, num_class):
		#calculate recall and precision score for each class
		recall = []
		precision = []
		true_false_matrix = np.zeros((num_class,num_class))
		for i in xrange(num_class):
			model.create_true_false_matrix(path[i], true_false_matrix, label = i)
			recall_label_i = true_false_matrix[i][i] / float(true_false_matrix.sum(axis = 0)[i])
			recall_label_i = str(round(recall_label_i * 100, 2)) + '%'
			recall.append(recall_label_i)

		for i in xrange(num_class):
			precision_label_i = true_false_matrix[i][i] / float(true_false_matrix.sum(axis = 1)[i])
			precision_label_i = str(round(precision_label_i * 100, 2)) + '%'
			precision.append(precision_label_i)
		for i in xrange(num_class):
			key = "Class " + str(i) + ' ' + str(path[i])
			self.recall_and_precision[key] = (recall[i], precision[i])


#test Vietnam News classification

model = NewsClassification()
path_train = ['./dataset/train/0. Giao duc/', './dataset/train/1. KH-CN/', \
				'./dataset/train/2. Phap luat/', './dataset/train/3. Suc khoe', \
	 			'./dataset/train/4. The thao', './dataset/train/5. Kinh te', \
	 			'./dataset/train/6. Van hoa - Giai tri','./dataset/train/7. Xa hoi', \
				'./dataset/train/8. Oto - Xe may']

path_test_all = ['./dataset/test/0. Giao duc/', './dataset/test/1. KH-CN/', \
		 		'./dataset/test/2. Phap luat/', './dataset/test/3. Suc khoe', './dataset/test/4. The thao', \
		 		'./dataset/test/5. Kinh te', './dataset/test/6. Van hoa - Giai tri', './dataset/test/7. Xa hoi', \
				'./dataset/test/8. Oto - Xe may']

path_test_class = [['./dataset/test/0. Giao duc/'], ['./dataset/test/1. KH-CN/'], \
		 		['./dataset/test/2. Phap luat/'], ['./dataset/test/3. Suc khoe'], ['./dataset/test/4. The thao'], \
		 		['./dataset/test/5. Kinh te'], ['./dataset/test/6. Van hoa - Giai tri'], ['./dataset/test/7. Xa hoi'], \
				['./dataset/test/8. Oto - Xe may']]


model.input_data(path_train, path_test_all)
training_size = model.train(path_train)

print "-----------------------------------------------------------"
print "Calculate Recall and Precision:"
num_class = len(path_train)
model.calc_recall_and_precision(path_test_class, num_class)
for key in model.recall_and_precision:
	print key, ":", model.recall_and_precision[key][0], "and", model.recall_and_precision[key][1] 

