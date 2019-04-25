# Text-Classification
[![](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/pandas-0.21.0-brightgreen.svg)](https://pypi.python.org/pypi/pandas/0.21.0)
[![](https://img.shields.io/badge/numpy-1.13.1-brightgreen.svg)](https://pypi.python.org/pypi/numpy/1.13.1)
[![](https://img.shields.io/badge/jieba-0.39-brightgreen.svg)](https://pypi.python.org/pypi/jieba/0.39)
[![](https://img.shields.io/badge/Keras-2.2.4-brightgreen.svg)](https://pypi.python.org/pypi/Keras/2.2.4)


## 项目介绍
通过对已有标签的文本进行训练，实现新文本的分类。<br>

## 更新说明
2019.3.25：项目最初是公司的一个舆情分析业务，后来参加了一些比赛又增加了一些小功能。当时只是想着把机器学习、深度学习的一些简单的模型整合在一起，锻炼一下工程能力。和一些网友交流后，觉得没必要搞一个通用型的模块（反正也没人用哈哈~）。最近刚好比较清闲，就本着越简单越好的目的把没啥用的花里胡哨的参数和函数都删了，只保留了预处理和卷积网络。

## 导入数据集:load_data
**准备了单一标签的电商数据4000多条和多标签的司法罪名数据15000多条，数据仅供学术研究使用，禁止商业传播。**<br>
* 单一标签的电商数据4000条为.csv格式，来源于真实电商评论，由'evaluation'和'label'两个字段组成，分别表示用户评论和正负面标签，建议pandas读取，读入后为dataframe。<br>
* 多标签的司法罪名数据15000条为.json格式，来源于2018‘法研杯’法律智能挑战赛（CAIL2018），由'fact'和'accusation'两个字段组成，分别表示事实陈述和罪名，读入后为列表。<br>
``` python
from TextClassification.load_data import load_data

# 单标签
data = load_data('single')
x = data['evaluation']
y = [[i] for i in data['label']]

# 多标签
data = load_data('multiple')
x = [i['fact'] for i in data]
y = [i['accusation'] for i in data]
```
![](https://github.com/renjunxiang/Text-Classification/blob/master/picture/data_single.png)
![](https://github.com/renjunxiang/Text-Classification/blob/master/picture/data_multiple.png)

## 文本预处理：DataPreprocess.py
**用于对原始文本数据做预处理，包含分词、转编码、长度统一等方法，已封装进TextClassification.py**<br>

``` python
preprocess = DataPreprocess()

# 处理文本
texts_cut = preprocess.cut_texts(texts, word_len)
preprocess.train_tokenizer(texts_cut, num_words)
texts_seq = preprocess.text2seq(texts_cut, sentence_len)

# 得到标签
preprocess.creat_label_set(labels)
labels = preprocess.creat_labels(labels)
```

## 模型训练及预测：TextClassification.py
**整合预处理、网络的训练、网络的预测，demo请参考两个demo脚本**<br>

**方法如下：**<br>
* fit：输入原始文本和标签，可以在已有的模型基础上继续训练，不输入模型则重新开始训练；<br>
* predict：输入原始文本；<br>

``` python
from TextClassification import TextClassification

clf = TextClassification()
texts_seq, texts_labels = clf.get_preprocess(x_train, y_train, 
                                             word_len=1, 
                                             num_words=2000, 
                                             sentence_len=50)
clf.fit(texts_seq=texts_seq,
        texts_labels=texts_labels,
        output_type=data_type,
        epochs=10,
        batch_size=64,
        model=None)

# 保存整个模块,包括预处理和神经网络
with open('./%s.pkl' % data_type, 'wb') as f:
    pickle.dump(clf, f)

# 导入刚才保存的模型
with open('./%s.pkl' % data_type, 'rb') as f:
    clf = pickle.load(f)
y_predict = clf.predict(x_test)
y_predict = [[clf.preprocess.label_set[i.argmax()]] for i in y_predict]
score = sum(y_predict == np.array(y_test)) / len(y_test)
print(score)  # 0.9288
```



