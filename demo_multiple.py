from TextClassification import load_data
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
import numpy as np

sess = tf.InteractiveSession()

# 导入数据
data_type = 'multiple'
data = load_data(data_type)
x = [i['fact'] for i in data]
y = [i['accusation'] for i in data]

# 拆分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

##### 以下是训练过程 #####

from TextClassification import TextClassification

clf = TextClassification()
texts_seq, texts_labels = clf.get_preprocess(x_train, y_train, word_len=1, num_words=2000, sentence_len=50)
clf.fit(texts_seq, texts_labels, data_type, 3, 64)

# 保存整个模块,包括预处理和神经网络
with open('./%s.pkl' % data_type, 'wb') as f:
    pickle.dump(clf, f)

##### 以下是预测过程 #####

# 导入刚才保存的模型
with open('./%s.pkl' % data_type, 'rb') as f:
    clf = pickle.load(f)
y_predict = clf.predict(x_test)
y_predict = clf.label2tag(y_predict, clf.preprocess.label_set)
score = sum([y_predict[i] == y_test[i] for i in range(len(y_predict))]) / len(y_predict)
print(score)  # 0.2136
