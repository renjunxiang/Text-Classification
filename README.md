# Text-Classification
[![](https://img.shields.io/badge/Python-3.5-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/pandas-0.21.0-brightgreen.svg)](https://pypi.python.org/pypi/pandas/0.21.0)
[![](https://img.shields.io/badge/numpy-1.13.1-brightgreen.svg)](https://pypi.python.org/pypi/numpy/1.13.1)
[![](https://img.shields.io/badge/jieba-0.39-brightgreen.svg)](https://pypi.python.org/pypi/jieba/0.39)
[![](https://img.shields.io/badge/gensim-3.2.0-brightgreen.svg)](https://pypi.python.org/pypi/gensim/3.2.0)
[![](https://img.shields.io/badge/Keras-2.1.5-brightgreen.svg)](https://pypi.python.org/pypi/Keras/2.1.5)
[![](https://img.shields.io/badge/scikit--learn-0.19.1-brightgreen.svg)](https://pypi.python.org/pypi/scikit-learn/0.19.1)

## 语言
Python3.5<br>
## 依赖库
pandas=0.21.0<br>
numpy=1.13.1<br>
jieba=0.39<br>
gensim=3.2.0<br>
scikit-learn=0.19.1<br>
keras=2.1.5<br>


## 项目介绍
通过对已有标签的文本进行训练，实现新文本的分类。<br>
目前完成了数据预处理、CNN、RNN、训练和预测的封装，后续会加入scikit-learn常用模型<br>

## 用法介绍
### 导入数据集:load_data
**准备了单一标签的电商数据4000多条和多标签的司法罪名数据15000多条，数据仅供学术研究使用，禁止商业传播。**<br>
* 单一标签的电商数据4000条为.csv格式，来源于真实电商评论，由'evaluation'和'label'两个字段组成，分别表示用户评论和正负面标签，建议pandas读取，读入后为dataframe。<br>
* 多标签的司法罪名数据15000条为.json格式，来源于2018‘法研杯’法律智能挑战赛（CAIL2018），由'fact'和'accusation'两个字段组成，分别表示事实陈述和罪名，读入后为列表。<br>
``` python
from TextClassification.load_data import load_data

#single target
data=load_data(name='single')
x=data['evaluation']
y=[[i] for i in data['label']]

#multiple target
data=load_data(name='multiple')
x=[i['fact'] for i in data]
y=[i['accusation'] for i in data]
```
![](https://github.com/renjunxiang/Text-Classification/blob/master/picture/data_single.png)
![](https://github.com/renjunxiang/Text-Classification/blob/master/picture/data_multiple.png)

### 文本预处理：DataPreprocess.py
**用于对原始文本数据做预处理，包含分词、转编码、长度统一等方法。**<br>
**方法如下：**<br>
* cut_texts：分词，输入文本、保留词语长度，输出词语列表<br>
* text2seq：词语列表转定长编码，输入词语列表，输出定长编码列表<br>
* text2vec：词语列表转词向量列表，输入词语列表，输出词向量列表<br>
* creat_label_set：创建标签集合，输入原始标签，输出不重复的标签列表<br>
* creat_labels：创建标签one-hot，输入原始标签、标签集合，输出one-hot的标签列表<br>
``` python
from TextClassification.DataPreprocess import DataPreprocess

process = DataPreprocess()
# cut texts
x_cut = process.cut_texts(texts=x, need_cut=True, word_len=2, savepath=None)
# texts to sequence
x_seq = process.text2seq(texts_cut=x_cut, tokenizer=None, tokenizer_savapah=None,
                         num_words=500, maxlen=20, batchsize=10000)
# list to array
x_seq = np.array(x_seq)

# texts to word vector					 
x_word_vec = process.text2vec(texts_cut=x, sg=1, size=128, window=5, min_count=1)
# texts vector
x_vec = process.text2vec(texts_cut=x, sg=1, size=128, window=5, min_count=1, merge=True)
```

### 模型训练及预测：TextClassification.py
**整合预处理、Keras神经网络、skleran机器学习的训练、预测，结果转标签，完整demo请参考demo文件夹**<br>
sklearn里面封装了SVC和LogisticRegression，中小型数据集表现要优于神经网络，要求标签为一维数组。<br>
神经网络封装了简单的CNN和RNN，要求标签为二维数组，从而可以转变为独热编码，标签可以多个。<br>
**方法如下：**<br>
* fit：整合预处理、模型训练，输入原始文本、转为编码的定长序列或者句向量。<br>
* predict：整合预处理、模型预测，输入原始文本、转为编码的定长序列或者句向量，model为None则调用训练的模型。<br>

``` python
from TextClassification import TextClassification

# neural network
model=TextClassification()
# train model
model.fit(x=X_train, y=y_train, method='CNN',model=None,
          x_need_preprocess=True, y_need_preprocess=True,
          epochs=10, batchsize=128, output_type='single')
# get label set
label_set=model.label_set
# predict data
y_predict=model.predict(x=X_test, x_need_preprocess=True)
# prediction to tag
y_predict_label=model.label2toptag(predictions=y_predict,labelset=label_set)
# calculate accuracy
print(sum([y_predict_label[i]==y_test[i] for i in range(len(y_predict))])/len(y_predict))

# sklearn
model.fit(x=X_train,
          y=y_train,
          x_need_preprocess=True,
          y_need_preprocess=False,
          method='SVM', output_type='single')

y_predict = model.predict(x=X_test, x_need_preprocess=True)
print(sum(y_predict == np.array(y_test)) / len(y_predict))
```



