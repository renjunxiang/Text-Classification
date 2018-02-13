from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
import pandas as pd
import jieba
from gensim.models import word2vec

jieba.setLogLevel('WARN')


def sentence_2_vec(train_data,
                   test_data=None,
                   size=5,
                   window=5,
                   min_count=1):
    '''
    
    :param train_data: 训练集
    :param test_data: 测试集
    :param size: 词向量维数
    :param window: word2vec滑窗大小
    :param min_count: word2vec滑窗内词语数量
    :param return: 返回词向量数组
    '''
    if test_data == None:
        train_data = [[word for word in jieba.lcut(sample) if word != ' '] for sample in train_data]
        model = word2vec.Word2Vec(train_data, size=size, window=window, min_count=min_count)
        train_data = [[model[word] for word in sample] for sample in train_data]
        return train_data

    else:
        train_data = [[word for word in jieba.lcut(sample) if word != ' '] for sample in train_data]
        test_data = [[word for word in jieba.lcut(sample) if word != ' '] for sample in test_data]
        data = train_data + test_data
        model = word2vec.Word2Vec(data, size=size, window=window, min_count=min_count)
        train_data = [[model[word] for word in sample] for sample in train_data]
        test_data = [[model[word] for word in sample] for sample in test_data]
        return train_data, test_data


if __name__ == '__main__':
    train_data = ['全面从严治党',
                  '国际公约和国际法',
                  '中国航天科技集团有限公司']
    test_data = ['全面从严测试']
    train_data_vec, test_data_vec = sentence_2_vec(train_data=train_data,
                                                   test_data=test_data,
                                                   size=5,
                                                   window=5,
                                                   min_count=1)
    print(train_data[0],'\n', pd.DataFrame(train_data_vec[0]),
          '\n',test_data[0],'\n', pd.DataFrame(test_data_vec[0]))
