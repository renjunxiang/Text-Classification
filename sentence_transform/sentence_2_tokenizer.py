import pandas as pd
import jieba
from keras.preprocessing.text import Tokenizer

jieba.setLogLevel('WARN')


def sentence_2_tokenizer(train_data,
                         test_data=None,
                         num_words=None,
                         word_index=False):
    '''
    
    :param train_data: 训练集
    :param test_data: 测试集
    :param num_words: 词库大小,None则依据样本自动判定
    :param word_index: 是否需要索引
    :param return: 返回编码后数组
    '''
    train_data = [' '.join([word for word in jieba.lcut(sample) if word != ' ']) for sample in train_data]
    test_data = [' '.join([word for word in jieba.lcut(sample) if word != ' ']) for sample in test_data]
    data = train_data + test_data
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(data)
    train_data = tokenizer.texts_to_sequences(train_data)
    test_data = tokenizer.texts_to_sequences(test_data)

    if word_index == False:
        if test_data == None:
            return train_data

        else:
            return train_data, test_data
    else:
        if test_data == None:
            return train_data, tokenizer.word_index

        else:
            return train_data, test_data, tokenizer.word_index


if __name__ == '__main__':
    train_data = ['全面从严治党',
                  '国际公约和国际法',
                  '中国航天科技集团有限公司']
    test_data = ['全面从严测试']
    train_data_vec, test_data_vec, word_index = sentence_2_tokenizer(train_data=train_data,
                                                                     test_data=test_data,
                                                                     num_words=None,
                                                                     word_index=True)
    print(train_data, '\n', train_data_vec, '\n',
          test_data[0], '\n', test_data_vec, '\n',
          'word_index\n',word_index)
