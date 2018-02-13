from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sentence_transform.sentence_2_sparse import sentence_2_sparse
from sentence_transform.sentence_2_vec import sentence_2_vec


def sklearn_supervised(language='English',
                       model_exist=False,
                       model_path=None,
                       model_name='SVM',
                       vector=True,
                       hashmodel='CountVectorizer',
                       savemodel=False,
                       train_dataset=None,
                       test_data=None):
    '''
    :param language: 语种,中文将调jieba先分词
    :param model_exist: 模型是否存在
    :param model_path: 模型路径
    :param model_name: 机器学习分类模型,SVM,KNN,Logistic
    :param hashmodel: 哈希方式:CountVectorizer,TfidfTransformer,HashingVectorizer
    :param savemodel: 保存模型
    :param train_dataset: 训练集[[数据],[标签]]
    :param test_data: 测试集[数据]
    :param return: 预测结果的数组
    '''
    if vector == True:
        train_data_transform, test_data_transform = sentence_2_vec(train_data=train_dataset[0],
                                                                   test_data=test_data,
                                                                   size=50,
                                                                   window=5,
                                                                   min_count=1)
        train_data_transform = [sum(i) / len(i) for i in train_data_transform]
        test_data_transform = [sum(i) / len(i) for i in test_data_transform]
    else:
        train_data_transform, test_data_transform = sentence_2_sparse(train_data=train_dataset[0],
                                                                      test_data=test_data,
                                                                      language=language,
                                                                      hash=True,
                                                                      hashmodel=hashmodel)
    train_label = train_dataset[1]
    model_path = model_path
    if model_exist == False:  # 如果不存在模型,调训练集训练
        model_name = model_name
        if model_name == 'KNN':
            model = KNeighborsClassifier(n_neighbors=min(len(train_label), 5))  # 调用KNN,近邻=5
            model.fit(train_data_transform, train_label)
        elif model_name == 'SVM':
            model = SVC(kernel='linear', C=1.0)  # 核函数为线性,惩罚系数为1
            model.fit(train_data_transform, train_label)
        elif model_name == 'Logistic':
            model = LogisticRegression(solver='liblinear', C=1.0)  # 核函数为线性,惩罚系数为1
            model.fit(train_data_transform, train_label)

        if savemodel == True:
            joblib.dump(model, model_path)  # 保存模型
    else:  # 存在模型则直接调用
        model = joblib.load(model_path)
    result = model.predict(test_data_transform)  # 对测试集进行预测
    return result


if __name__ == '__main__':
    print('example:English')
    train_dataset = [['he likes apple',
                      'he really likes apple',
                      'he hates apple',
                      'he really hates apple'],
                     ['possitive', 'possitive', 'negative', 'negative']]
    print('train data\n',
          pd.DataFrame({'data': train_dataset[0],
                        'label': train_dataset[1]},
                       columns=['data', 'label']))
    test_data = ['she likes apple',
                 'she really hates apple',
                 'tom likes apple',
                 'tom really hates apple'
                 ]
    test_label = ['possitive', 'negative', 'possitive', 'negative']

    result = sklearn_supervised(train_dataset=train_dataset,
                                test_data=test_data,
                                model_name='SVM',
                                language='English')
    print('score:', np.sum(result == np.array(test_label)) / len(result))
    result = pd.DataFrame({'data': test_data,
                           'label': test_label,
                           'predict': result},
                          columns=['data', 'label', 'predict'])
    print('test\n', result)

    print('example:Chinese')
    train_dataset = [['国王喜欢吃苹果',
                      '国王非常喜欢吃苹果',
                      '国王讨厌吃苹果',
                      '国王非常讨厌吃苹果'],
                     ['正面', '正面', '负面', '负面']]
    print('train data\n',
          pd.DataFrame({'data': train_dataset[0],
                        'label': train_dataset[1]},
                       columns=['data', 'label']))
    test_data = ['涛哥喜欢吃苹果',
                 '涛哥讨厌吃苹果',
                 '涛哥非常喜欢吃苹果',
                 '涛哥非常讨厌吃苹果']
    test_label = ['正面', '负面', '正面', '负面']
    result = sklearn_supervised(train_dataset=train_dataset,
                                test_data=test_data,
                                model_name='SVM',
                                language='Chinese')
    print('score:', np.sum(result == np.array(test_label)) / len(result))
    result = pd.DataFrame({'data': test_data,
                           'label': test_label,
                           'predict': result},
                          columns=['data', 'label', 'predict'])
    print('test\n', result)
