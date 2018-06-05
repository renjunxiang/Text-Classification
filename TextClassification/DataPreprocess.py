import json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import word2vec
import jieba
import pickle

jieba.setLogLevel('WARN')


class DataPreprocess():
    def __init__(self):
        self.texts_cut = None
        self.tokenizer = None
        self.tokenizer_fact = None

    def cut_texts(self, texts=None, need_cut=True, word_len=1, savepath=None):
        '''
        Use jieba to cut texts
        :param texts:list of texts
        :param need_cut:whether need cut text
        :param word_len:min length of words to keep,in order to delete stop-words
        :param savepath:path to save word list in json file
        :return:
        '''
        if need_cut:
            if word_len > 1:
                texts_cut = [[word for word in jieba.lcut(text) if len(word) >= word_len] for text in texts]
            else:
                texts_cut = [jieba.lcut(one_text) for one_text in texts]
        else:
            if word_len > 1:
                texts_cut = [[word for word in text if len(word) >= word_len] for text in texts]
            else:
                texts_cut = texts

        if savepath is not None:
            with open(savepath, 'w') as f:
                json.dump(texts_cut, f)
        return texts_cut

    def text2seq(self, texts_cut=None, tokenizer=None, tokenizer_savapah=None,
                 num_words=2000, maxlen=30, batchsize=10000):
        '''
        文本转序列，用于神经网络的ebedding层输入。训练集过大全部转换会内存溢出，每次放10000个样本
        :param texts_cut: 分词后的文本列表
        :param tokenizer:转换字典，keras的一个方法
        :param tokenizer_savapah:字典保存路径
        :param num_words:字典保留的高频词数量
        :param maxlen:保留长度
        :param batchsize:每次参与提取的文档数
        :return:向量列表
        eg. ata_transform.text2seq(texts_cut=train_fact_cut,num_words=2000, maxlen=500)
        '''
        texts_cut_len = len(texts_cut)

        if tokenizer is None:
            tokenizer = Tokenizer(num_words=num_words)
            n = 0
            # 分批训练
            while n < texts_cut_len:
                tokenizer.fit_on_texts(texts=texts_cut[n:n + batchsize])
                n += batchsize
                if n < texts_cut_len:
                    print('tokenizer finish fit %d samples' % n)
                else:
                    print('tokenizer finish fit %d samples' % texts_cut_len)
            self.tokenizer = tokenizer

        if tokenizer_savapah:
            with open(tokenizer_savapah, mode='wb') as f:
                pickle.dump(tokenizer, f)

        # 全部转为数字序列
        fact_seq = tokenizer.texts_to_sequences(texts=texts_cut)
        print('finish texts to sequences')

        # 内存不够，删除
        del texts_cut

        n = 0
        fact_pad_seq = []
        # 分批执行pad_sequences
        while n < texts_cut_len:
            fact_pad_seq += list(pad_sequences(fact_seq[n:n + 10000], maxlen=maxlen,
                                               padding='post', value=0, dtype='int'))
            n += 10000
            if n < texts_cut_len:
                print('finish pad sequences %d/%d' % (n, texts_cut_len))
            else:
                print('finish pad sequences %d/%d' % (texts_cut_len, texts_cut_len))
        return fact_pad_seq

    def text2vec(self, texts_cut=None, model_word2vec=None,
                 word2vec_savepath=None, word2vec_loadpath=None,
                 sg=1, size=128, window=5, min_count=1):
        '''
        文本的词语序列转为词向量序列，可以用于机器学习或者深度学习
        :param texts_cut: 词语序列
        :param model_word2vec: word2vec的模型
        :param word2vec_savepath: word2vec保存路径
        :param word2vec_loadpath: word2vec导入路径
        :param sg: 0 CBOW,1 skip-gram
        :param size: the dimensionality of the feature vectors
        :param window: the maximum distance between the current and predicted word within a sentence
        :param min_count: ignore all words with total frequency lower than this 
        :return: 
        '''
        if model_word2vec is None:
            if word2vec_loadpath:
                model_word2vec = word2vec.Word2Vec.load(word2vec_loadpath)
            else:
                model_word2vec = word2vec.Word2Vec(texts_cut, sg=sg, size=size, window=window, min_count=min_count)
        if word2vec_savepath:
            model_word2vec.save(word2vec_savepath)

        return [[model_word2vec[word] for word in text_cut if word in model_word2vec] for text_cut in texts_cut]

    def creat_label_set(self, labels):
        '''
        获取标签集合，用于one-hot
        :param labels: 原始标签集
        :return:
        '''
        label_set = []
        for i in labels:
            label_set += i
        return np.array(list(set(label_set)))

    def creat_label(self, label, label_set):
        '''
        构建标签one-hot
        :param label: 原始标签
        :param label_set: 标签集合
        :return: 标签one-hot形式的array
        eg. creat_label(label=data_valid_accusations[12], label_set=accusations_set)
        '''
        label_zero = np.zeros(len(label_set))
        label_zero[np.in1d(label_set, label)] = 1
        return label_zero

    def creat_labels(self, labels=None, label_set=None):
        '''
        调用creat_label遍历标签列表生成one-hot二维数组
        :param label: 原始标签集
        :param label_set: 标签集合
        :return:
        '''
        labels_one_hot = list(map(lambda x: self.creat_label(label=x, label_set=label_set), labels))
        return labels_one_hot
