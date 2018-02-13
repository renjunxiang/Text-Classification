import numpy as np
import pandas as pd
from models.sklearn_supervised import sklearn_supervised
from sklearn.model_selection import train_test_split

# 读取正负面标签数据
positive = pd.read_excel('D:/github/Text-Classification/data/demo_score/data.xlsx',
                         sheet_name='positive')
negative = pd.read_excel('D:/github/Text-Classification/data/demo_score/data.xlsx',
                         sheet_name='negative')
# 分隔训练集和测试集
total = pd.concat([positive, negative], axis=0)
X_train, X_test, y_train, y_test = train_test_split(total.loc[:, 'evaluation'],
                                                    total.loc[:, 'label'],
                                                    test_size=0.33,
                                                    random_state=42)
model_name = ['SVM', 'KNN', 'Logistic']
hashmodel = ['CountVectorizer', 'TfidfTransformer', 'HashingVectorizer']
score_all = []

for i in model_name:
    for j in hashmodel:
        print('model name:', i, '; hashmodel:', j)
        result = sklearn_supervised(language='Chinese',
                                    model_exist=False,
                                    model_path=None,
                                    model_name=i,
                                    hashmodel=j,
                                    savemodel=False,
                                    train_dataset=[list(X_train), list(y_train)],
                                    test_data=list(X_test))
        score = np.sum(result == np.array(y_test)) / len(result)
        score_all.append([i, j, score])

for i in model_name:
    print('model name:', i, '; hashmodel:', None)
    result = sklearn_supervised(language='Chinese',
                                model_exist=False,
                                model_path=None,
                                model_name=i,
                                hashmodel=None,
                                vector=True,
                                savemodel=False,
                                train_dataset=[list(X_train), list(y_train)],
                                test_data=list(X_test))
    score = np.sum(result == np.array(y_test)) / len(result)
    score_all.append([i, 'vector', score])
# 预测结果得分矩阵
score_all = pd.DataFrame(score_all,
                         columns=['model_name', 'Transformer', 'score'])
score_all = pd.pivot_table(data=score_all,
                           index='model_name',
                           columns='Transformer',
                           values='score')
print('score_all:\n', score_all)

# predict_evaluate = pd.DataFrame({'document': X_test,
#                                  'label': y_test,
#                                  'predict': result},
#                                 columns=['document', 'label', 'predict'])
# predict_evaluate=predict_evaluate.reset_index(drop=True)
# predict_evaluate_wrong=predict_evaluate.loc[predict_evaluate.loc[:,'label']!=
#                                             predict_evaluate.loc[:,'predict'],:]
# predict_evaluate_wrong.to_excel('D:/github/Text-Classification/data/demo_score/predict.xlsx',
#                                 index=False)#分类错误的数据保存下来

result = sklearn_supervised(language='Chinese',
                            model_exist=False,
                            model_path=None,
                            model_name='SVM',
                            hashmodel=None,
                            vector=True,
                            savemodel=False,
                            train_dataset=[list(X_train), list(y_train)],
                            test_data=list(X_test))
predict_evaluate = pd.DataFrame({'document': X_test,
                                 'label': y_test,
                                 'predict': result},
                                columns=['document', 'label', 'predict'])
predict_evaluate = predict_evaluate.reset_index(drop=True)
predict_evaluate_wrong = predict_evaluate.loc[predict_evaluate.loc[:, 'label'] !=
                                              predict_evaluate.loc[:, 'predict'], :]
predict_evaluate.to_excel('D:/github/Text-Classification/data/demo_score/predict.xlsx',
                          index=False)  # 分类错误的数据保存下来
predict_evaluate_wrong.to_excel('D:/github/Text-Classification/data/demo_score/predict_wrong.xlsx',
                                index=False)  # 分类错误的数据保存下来
