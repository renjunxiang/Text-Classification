import numpy as np
import pandas as pd
from models.sklearn_supervised import sklearn_supervised
from sklearn.model_selection import train_test_split
from testdata import config

# 读取正负面标签数据
positive = pd.read_excel(config.rawdata['positive']['path'],
                         sheet_name=config.rawdata['positive']['sheetname'])
negative = pd.read_excel(config.rawdata['negative']['path'],
                         sheet_name=config.rawdata['negative']['sheetname'])

update_v1 = pd.read_excel(config.test['20180130']['update']['path'],
                          sheet_name=config.test['20180130']['update']['sheetname'])

test_data = pd.read_excel(config.test['20180130']['test']['path'],
                          sheet_name=config.test['20180130']['test']['sheetname'])
# 分隔训练集和测试集
total = pd.concat([positive, negative, update_v1], axis=0)
X_train, X_test, y_train, y_test = train_test_split(total.loc[:, 'evaluation'],
                                                    total.loc[:, 'label'],
                                                    test_size=0.33,
                                                    random_state=42)

result = sklearn_supervised(language='Chinese',
                            model_exist=False,
                            model_path=None,
                            model_name='SVM',
                            hashmodel=None,
                            vector=True,
                            savemodel=False,
                            train_dataset=[list(X_train), list(y_train)],
                            test_data=list(X_test))

score = np.sum(result == np.array(y_test)) / len(result)

result1=pd.DataFrame({'predict':result,'label':list(y_test),'1':1})
score_all = pd.pivot_table(data=result1,
                           index='predict',
                           columns='label',
                           values='1',aggfunc='count')/len(result)
###############################################################################
result = sklearn_supervised(language='Chinese',
                            model_exist=False,
                            model_path=None,
                            model_name='SVM',
                            hashmodel=None,
                            vector=True,
                            savemodel=False,
                            train_dataset=[total.loc[:, 'evaluation'], total.loc[:, 'label']],
                            test_data=list(test_data.iloc[:, 0]))
predict_evaluate = pd.DataFrame({'evaluation': test_data.iloc[:, 0],
                                 'predict': result},
                                columns=['evaluation', 'predict'])
predict_evaluate = predict_evaluate.reset_index(drop=True)
predict_evaluate.to_excel(config.filename + '20180130/predict_v1.xlsx',
                          index=False)  # 分类错误的数据保存下来
