from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
import pandas as pd
import jieba

jieba.setLogLevel('WARN')


def sentence_2_sparse(train_data,
                      test_data=None,
                      language='Chinese',
                      hash=True,
                      hashmodel='CountVectorizer'):
    '''
    
    :param train_data: 训练集
    :param test_data: 测试集
    :param language: 语种
    :param hash: 是否转哈希存储
    :param hashmodel: 哈希计数的方式
    :param return: 返回编码后稀疏矩阵
    '''
    # 分词转one-hot dataframe
    if test_data==None:
        if hash == False:
            train_data = pd.DataFrame([pd.Series([word for word in jieba.lcut(sample) if word != ' ']).value_counts()
                                       for sample in train_data]).fillna(0)
        # 中文需要先分词空格分隔,再转稀疏矩阵
        else:
            if language == 'Chinese':
                train_data = [' '.join([word for word in jieba.lcut(sample) if word != ' ']) for sample in train_data]
            if hashmodel == 'CountVectorizer':  # 只计数
                count_train = CountVectorizer()
                train_data_hashcount = count_train.fit_transform(train_data)  # 训练数据转哈希计数
            elif hashmodel == 'TfidfTransformer':  # 计数后计算tf-idf
                count_train = CountVectorizer()
                train_data_hashcount = count_train.fit_transform(train_data)  # 训练数据转哈希计数
                tfidftransformer = TfidfTransformer()
                train_data_hashcount = tfidftransformer.fit(train_data_hashcount).transform(train_data_hashcount)
            elif hashmodel == 'HashingVectorizer':  # 哈希计算
                vectorizer = HashingVectorizer(stop_words=None, n_features=10000)
                train_data_hashcount = vectorizer.fit_transform(train_data)  # 训练数据转哈希后的特征,避免键值重叠导致过大有一个计算的
            return train_data_hashcount
        return train_data
    else:
        # 中文需要先分词空格分隔,再转稀疏矩阵,如果包含测试集,测试集转hash需要在训练集的词库基础上执行
        if language == 'Chinese':
            train_data = [' '.join([word for word in jieba.lcut(sample) if word != ' ']) for sample in train_data]
            test_data = [' '.join([word for word in jieba.lcut(sample) if word != ' ']) for sample in test_data]
        if hashmodel == 'CountVectorizer':  # 只计数
            count_train = CountVectorizer()
            train_data_hashcount = count_train.fit_transform(train_data)  # 训练数据转哈希计数
            count_test = CountVectorizer(vocabulary=count_train.vocabulary_)  # 测试数据调用训练词库
            test_data_hashcount = count_test.fit_transform(test_data)  # 测试数据转哈希计数
        elif hashmodel == 'TfidfTransformer':  # 计数后计算tf-idf
            count_train = CountVectorizer()
            train_data_hashcount = count_train.fit_transform(train_data)  # 训练数据转哈希计数
            count_test = CountVectorizer(vocabulary=count_train.vocabulary_)  # 测试数据调用训练词库
            test_data_hashcount = count_test.fit_transform(test_data)  # 测试数据转哈希计数
            tfidftransformer = TfidfTransformer()
            train_data_hashcount = tfidftransformer.fit(train_data_hashcount).transform(train_data_hashcount)
            test_data_hashcount = tfidftransformer.fit(test_data_hashcount).transform(test_data_hashcount)
        elif hashmodel == 'HashingVectorizer':  # 哈希计算
            vectorizer = HashingVectorizer(stop_words=None, n_features=10000)
            train_data_hashcount = vectorizer.fit_transform(train_data)  # 训练数据转哈希后的特征,避免键值重叠导致过大有一个计算的
            test_data_hashcount = vectorizer.fit_transform(test_data)  # 测试数据转哈希后的特征
        return train_data_hashcount, test_data_hashcount


if __name__ == '__main__':
    train_data = ['全面从严治党',
                  '国际公约和国际法',
                  '中国航天科技集团有限公司']
    test_data = ['全面从严测试']
    print('train_data\n',train_data,'\ntest_data\n',test_data)
    print('sentence_2_sparse(train_data=train_data,hash=False)\n',
          sentence_2_sparse(train_data=train_data, hash=False))
    print('sentence_2_sparse(train_data=train_data,hash=True)\n',
          sentence_2_sparse(train_data=train_data, hash=True))
    m,n=sentence_2_sparse(train_data=train_data, test_data=test_data, hash=True)
    print('sentence_2_sparse(train_data=train_data,test_data=test_data,hash=True)\n',
          'train_data\n',m,'\ntest_data\n',n)
