#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import dictparser as dp
import os
import numpy as np
from underthesea import word_sent
from scipy.sparse import coo_matrix
import math


class NewsData:
	def __init__(self):
		self.news_dict = dp.DictParser()
		self.file_id = -1 #start from 0

	def create_new_dict(self, path):
		cnt = 1
		for i in path:
			for file_name in os.listdir(i):
				file_name = os.path.join(i, file_name)
				with open(file_name) as data_file:
					data = json.load(data_file)
					string = data["content"]
					self.news_dict.parse_string_to_arr_id(string, cnt)
					cnt += 1
		return self.news_dict.get_map_word_to_id()
	
	def load_dict_from_file(self, path):
		return self.news_dict.load_dict(path)

	def write_dict_to_file(self, dict):
		content = ""
		for key in dict.keys():
			if key == "__unknow":
				continue
			content += key + '\n'
		with open('./dataset/news_dict.txt', 'w') as f:
			f.write(content.encode('utf-8'))

	def create_freq_words(self, string):
		map = {}
		list_word = word_sent(string)
		for word in list_word:
			id = 0
			id = self.news_dict.load_word_id(word)
			if map.has_key(id):
				map[id] += 1
			else:
				map[id] = 1
		return (map, len(list_word))

	def calc_tf_idf(self, map, cnt_word, idf, features):
		for key in map.keys():
			tmp = []
			if idf.has_key(key):
				idf[key] += 1
			else:
				idf[key] = 1
			tf = map[key] / float(cnt_word)
			tmp = [self.file_id, key, map[key], tf]
			features.append(tmp)
		return (idf, features)

	def create_features_and_labels(self, path, label_id):
		features = []
		labels = []
		text_cnt = 0
		idf = {}
		for file_name in os.listdir(path):
			file_name = os.path.join(path, file_name)
			with open(file_name) as data_file:
				data = json.load(data_file)
				string = data["content"]
				if string == "":
					continue
				self.file_id += 1
				text_cnt += 1
				(map, cnt_word) = self.create_freq_words(string)
				(idf, features) = self.calc_tf_idf(map, cnt_word, idf, features)
		tmp = (text_cnt) * [label_id]
		labels.extend(tmp)
		for key in idf.keys():
			idf[key] = math.log(text_cnt / float(idf[key]))

		#remove stop words using Tfidf
		for i in features:
			i[3] = i[3] * idf[i[1]]
			if i[3] < 0.002:
				i[2] = 0
		return (features, labels)

	def create_matrix_bag_of_words(self, path, nwords, label):
		features = []
		labels = []
		for folder in path:
			(tmp_features, tmp_labels) = self.create_features_and_labels(folder, label)
			features.extend(tmp_features)
			labels.extend(tmp_labels)
			label += 1
		features = np.array(features)
		features = coo_matrix((features[:, 2], \
				 (features[:, 0], features[:, 1])), shape = (len(labels), nwords))
		self.file_id = -1 #restore file_id to create other matrix bag of words
		return(features, labels)