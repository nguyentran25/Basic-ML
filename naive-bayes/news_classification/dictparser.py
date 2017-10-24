#!/usr/bin/env python
# -*- coding: utf-8 -*-

from underthesea import word_sent

class DictParser:

    def __init__(self):
        self._word_unknow = "__unknow"
        self.map_word_to_id = {}
        self.arr_id_to_word = []
        self.count_word = 0
        self.load_or_create_word_id(self._word_unknow)
        self.map_label_to_id = {}
        self.arr_id_to_label = []
        self.count_label = 0

    # word
    def load_dict(self, path_to_file):
        with open(path_to_file) as f:
            mylist = f.read().splitlines()
        for i in mylist:
            item = unicode(i, "utf-8")
            self.load_or_create_word_id(item)
        return self.map_word_to_id


    def get_size_dict(self):
        return self.count_word

    def get_map_word_to_id(self):
        return self.map_word_to_id

    def get_arr_id_to_word(self):
        return self.arr_id_to_word

    def load_word_id(self, word):
        word = word.lower()
        if self.map_word_to_id.__contains__(word) == False:
            return self.map_word_to_id[self._word_unknow]
        else:
            return self.map_word_to_id[word]

    def load_id_word(self, id):
        if id < self.arr_id_to_word.__len__():
            return self.arr_id_to_word[id]
        else:
            return "__unknow"
    def load_or_create_word_id(self, word):
        word = word.lower()
        if self.map_word_to_id.__contains__(word) == False:
            self.arr_id_to_word.append(word)
            self.map_word_to_id[word] = self.count_word
            self.count_word += 1
            return self.count_word - 1
        else :
            return self.map_word_to_id[word]

    def parse_string_to_arr_id(self, string_input, cnt):
        arr_id_ret = []
        items = word_sent(string_input)
        with open('./data/stop') as f:
            stop_word = f.readlines()
            stop_word = [x.strip() for x in stop_word]
        items  = [word for word in items if word.lower() not in stop_word]
        # load or create word
        for (i, item) in enumerate(items):
            arr_id_ret.append(self.load_or_create_word_id(item))
        print "File", cnt, "done"
        return arr_id_ret

    def parse_arr_id_to_string(self, arr_word_id):
        string_ret = ""
        # load or create word
        for (i, item) in enumerate(arr_word_id):
            string_ret += " "+self.load_id_word(item)
        return string_ret

    # label
    def get_size_label(self):
        return self.count_label

    def get_map_label_to_id(self):
        return self.map_label_to_id

    def get_arr_id_to_label(self):
        return self.arr_id_to_label

    def load_or_create_label_id(self, label):
        label = label.lower()
        if self.map_label_to_id.__contains__(label) == False:
            self.arr_id_to_label.append(label)
            self.map_label_to_id[label] = self.count_label
            self.count_label += 1
            return self.count_label - 1
        else:
            return self.map_label_to_id[label]

    def load_label_id(self, label):
        label = label.lower()
        if self.map_label_to_id.__contains__(label) == False:
            return None
        else:
            return self.map_label_to_id[label]

    def load_id_label(self, id):
        if id < self.arr_id_to_label.__len__():
            return self.arr_id_to_label[id]
        else:
            return self._word_unknow
