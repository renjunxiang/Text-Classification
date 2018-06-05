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
### 导入数据集 load_data
准备了单一标签的电商数据6000条和多标签的司法罪名数据15000条。<br>
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
**包含分词、转编码、长度统一。<br>
``` python
from DataPreprocess import DataPreprocess

process = DataPreprocess()
# cut texts
x_cut = process.cut_texts(texts=x, need_cut=True, word_len=2, savepath=None)
# use average length
if maxlen is None:
    maxlen = int(np.array([len(x) for i in x_cut]).mean())
# texts to sequence
x_seq = process.text2seq(texts_cut=x_cut, tokenizer=tokenizer, tokenizer_savapah=None,
                         num_words=num_words, maxlen=maxlen, batchsize=10000)
x_seq = np.array(x_seq)
```

### 模型训练及预测：TextClassification.py
已包含预处理以及Keras神经网络训练、预测，结果转标签<br>
``` python
from TextClassification import TextClassification

model=TextClassification()
model.fit(x=X_train, y=y_train, method='CNN',model=None,
          x_need_preprocess=True, y_need_preprocess=True,
          epochs=10, batchsize=128, output_type='single')
label_set=model.label_set
y_predict=model.predict(x=X_test, x_need_preprocess=True)
y_predict_label=model.label2toptag(predictions=y_predict,labelset=label_set)
print(sum([y_predict_label[i]==y_test[i] for i in range(len(y_predict))])/len(y_predict))
```



