import numpy as np
import pandas as pd
from supervised_classify import supervised_classify
from sklearn.model_selection import train_test_split

positive=pd.read_excel('D:/github/machine-learning/NLP/data/demo_score/data.xlsx',
                   sheet_name='positive')
negative=pd.read_excel('D:/github/machine-learning/NLP/data/demo_score/data.xlsx',
                   sheet_name='negative')

total=pd.concat([positive,negative],axis=0)
X_train, X_test, y_train, y_test = train_test_split(total.loc[:, 'evaluation'],
                                                    total.loc[:, 'label'],
                                                    test_size=0.33,
                                                    random_state=42)
result = supervised_classify(language='Chinese',
                             model_exist=False,
                             model_path='D:/github/machine-learning/NLP/data/demo_score/model.m',
                             model_name='SVM',
                             vector=True,
                             hashmodel='CountVectorizer',
                             savemodel=True,
                             train_dataset=[list(X_train), list(y_train)],
                             test_data=list(X_test))
print('score:', np.sum(result == np.array(y_test)) / len(result))
