import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

jieba.setLogLevel('WARN')


class DataPreprocess():
    def __init__(self, tokenizer=None,
                 label_set=None):
        self.tokenizer = tokenizer
        self.num_words = None
        self.label_set = label_set
        self.sentence_len = None
        self.word_len = None

    def cut_texts(self, texts=None, word_len=1):
        """
        对文本分词
        :param texts: 文本列表
        :param word_len: 保留最短长度的词语
        :return:
        """
        if word_len > 1:
            texts_cut = [[word for word in jieba.lcut(text) if len(word) >= word_len] for text in texts]
        else:
            texts_cut = [jieba.lcut(one_text) for one_text in texts]

        self.word_len = word_len

        return texts_cut

    def train_tokenizer(self,
                        texts_cut=None,
                        num_words=2000):
        """
        生成编码字典
        :param texts_cut: 分词的列表
        :param num_words: 字典按词频从高到低保留数量
        :return:
        """
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(texts=texts_cut)
        num_words = min(num_words, len(tokenizer.word_index) + 1)
        self.tokenizer = tokenizer
        self.num_words = num_words

    def text2seq(self,
                 texts_cut,
                 sentence_len=30):
        """
        文本转序列，用于神经网络的ebedding层输入。
        :param texts_cut: 分词后的文本列表
        :param sentence_len: 文本转序列保留长度
        :return:sequence list
        """
        tokenizer = self.tokenizer
        texts_seq = tokenizer.texts_to_sequences(texts=texts_cut)
        del texts_cut

        texts_pad_seq = pad_sequences(texts_seq,
                                      maxlen=sentence_len,
                                      padding='post',
                                      truncating='post')
        self.sentence_len = sentence_len
        return texts_pad_seq

    def creat_label_set(self, labels):
        '''
        获取标签集合，用于one-hot
        :param labels: 原始标签集
        :return:
        '''
        label_set = set()
        for i in labels:
            label_set = label_set.union(set(i))

        self.label_set = np.array(list(label_set))

    def creat_label(self, label):
        '''
        构建标签one-hot
        :param label: 原始标签
        :return: 标签one-hot形式的array
        eg. creat_label(label=data_valid_accusations[12], label_set=accusations_set)
        '''
        label_set = self.label_set
        label_zero = np.zeros(len(label_set))
        label_zero[np.in1d(label_set, label)] = 1
        return label_zero

    def creat_labels(self, labels=None):
        '''
        调用creat_label遍历标签列表生成one-hot二维数组
        :param label: 原始标签集
        :return:
        '''
        label_set = self.label_set
        labels_one_hot = [self.creat_label(label) for label in labels]

        return np.array(labels_one_hot)
