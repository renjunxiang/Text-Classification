from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np


def SklearnClf(method='SVM', **param):
    '''
    '''
    if method == 'SVM':
        model = SVC(**param)
    elif method == 'Logistic':
        model = LogisticRegression(**param)
    return model


if __name__ == '__main__':
    x = np.random.randint(1, 10, 20).reshape([5, 4])
    y = np.array([1, 2, 1, 2, 3])
    model = SklearnClf(model_name='SVM')
    model.fit(x, y)
    model.predict(x)
