from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import History
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transform.sentence_2_sparse import sentence_2_sparse
from sentence_transform.sentence_2_vec import sentence_2_vec
from models.neural_LSTM import neural_LSTM
from models.keras_log_plot import keras_log_plot

positive = pd.read_excel('D:/github/Text-Classification/data/demo_score/data.xlsx',
                         sheet_name='positive')
negative = pd.read_excel('D:/github/Text-Classification/data/demo_score/data.xlsx',
                         sheet_name='negative')
# 分隔训练集和测试集
total = pd.concat([positive, negative], axis=0)
# 转词向量
data_transform = sentence_2_vec(train_data=total.loc[:, 'evaluation'],
                                test_data=None,
                                size=5,
                                window=5,
                                min_count=1)
# 将不同长度的文本进行'截断/填充'至相同长度,不设置maxlen则填充至最长
data_transform = pad_sequences(data_transform, maxlen=None, padding='post', value=0, dtype='float32')
label_transform = np.array(pd.get_dummies(total.loc[:, 'label']))
print(data_transform.shape)
# 拆分为训练集和测试集
train_data, test_data, train_label, test_label = train_test_split(data_transform,
                                                                  label_transform,
                                                                  test_size=0.33,
                                                                  random_state=42)
model = neural_LSTM(input_shape=data_transform.shape[-2:],
                    net_shape=[64, 64, 128, 2],
                    optimizer_name='SGD',
                    lr=0.001)

history=History()
model.fit(train_data, train_label, batch_size=100, epochs=10, verbose=2,
          validation_data=(test_data, test_label), callbacks=[history])
train_log=pd.DataFrame(history.history)
keras_log_plot(train_log)

# model.save('ppp.h5')
# from keras.models import load_model
# model_new=load_model('ppp.h5')