# -*- coding: utf-8 -*-
#!/usr/bin/env python

import random
import numpy as np
from config import Config

PAD_token = 0
SOS_token = 1
EOS_token = 2


class WordDict:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        # PAD : padding
        # SOS : start of sentence
        # EOS : end of sentence
        self.index2word = {PAD_token: "<PAD>", SOS_token: "<SOS>", EOS_token: "<EOS>"}
        self.n_words = 3  # Count PAD, SOS and EOS

    def add_indexes(self, sentence):
        for word in sentence.split(' '):
            self.add_index(word)

    def add_index(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def sentence_to_indexes(self, sentence, max_length):
        indexes = [self.word2index[word] for word in sentence.split(' ')][:max_length - 1]
        indexes.append(EOS_token)
        n_indexes = len(indexes)
        indexes.extend([PAD_token for _ in range(max_length - len(indexes))])
        return indexes, n_indexes

    def indexes_to_sentence(self, indexes):
        indexes = filter(lambda i: i != PAD_token, indexes)
        indexes = map(lambda i: self.index2word[i], indexes)
        return ' '.join(indexes)


class Corpus:
    def __init__(self, dict, max_length, path):
        self.max_length = max_length
        # open().read()讀取整份文件
        # 將讀進來的string用UTF-8編碼，並經過filter_raw_string後再用換行切開，將資料存進lines
        self.lines = self.filter_raw_string(open(path,encoding='utf-8').read().encode('utf-8').decode('utf-8-sig')).split('\n')
        # 將lines中的每一行用tab切開變成 Error/Correction
        self.pairs = [[s for s in l.split('\t')] for l in self.lines]
        self.dict = dict
        count = 0
        # print(self.pairs[521])
        for pair in self.pairs:
            self.dict.add_indexes(pair[0])
            self.dict.add_indexes(pair[1])
            count = count+1
            # print(count)

    def filter_raw_string(self, str):
        # strip()將字串去除前後空格
        # 將字串去除 " <> "
        # return str.strip().translate(None, "<>")

        trantab = str.maketrans({'<' : '', '>' : ''})
        return str.strip().translate(trantab)
        # return str.strip()


    def next_batch(self, batch_size=5):
        # 從 pairs 中隨機取 batch_size 個 pair 作為下一次的輸入
        pairs = np.array(random.sample(self.pairs, batch_size))
        # 將 sample 過後的 pairs[i,0] 當成 input
        input_lens = [self.dict.sentence_to_indexes(s, self.max_length) for s in pairs[:, 0]]
        # 將 sample 過後的 pairs[i,1] 當成 target
        target_lens = [self.dict.sentence_to_indexes(s, self.max_length) for s in pairs[:, 1]]

        input_lens, target_lens = zip(*sorted(zip(input_lens, target_lens), key=lambda p: p[0][1], reverse=True))
        # 在 python3 中加上 list 將 map 轉回 sequence data
        inputs = list(map(lambda i: i[0], input_lens))
        len_inputs = list(map(lambda i: i[1], input_lens))
        targets = list(map(lambda i: i[0], target_lens))
        len_targets = list(map(lambda i: i[1], target_lens))
        return inputs, targets, len_inputs, len_targets


def build_corpus():
    word_dict = WordDict()
    train_corpus = Corpus(word_dict, Config.max_seq_length, Config.train_data_path)
    eval_corpus = Corpus(word_dict, Config.max_seq_length, Config.eval_data_path)
    return train_corpus, eval_corpus, word_dict