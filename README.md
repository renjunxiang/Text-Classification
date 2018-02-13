# Text-Classification

## 语言
Python3.5<br>
## 依赖库
requests=2.18.4<br>
baidu-aip=2.1.0.0<br>
pandas=0.21.0<br>
numpy=1.13.1<br>
jieba=0.39<br>
gensim=3.2.0<br>
scikit-learn=0.19.1<br>
keras=2.1.1<br>





## 项目介绍
通过对已有标签的帖子进行训练，实现新帖子的情感分类。现阶段通过第三方购买的数据，文本为爬虫抓取的电商购物评论，标签为“正面/负面”。<br>
目前传统的机器学习模型准确率在84%左右，SVM效果最好，但深度学习方法里面LSTM效果较差仅为64%，一维卷积为81%，查找原因中~<br>
***PS：这是本人毕业后通过业余时间做的第一个项目，结合了大半年时间的所学，耗时一个月时间已近尾声~目前公司情感分析也参考了这个项目，有很多不足，欢迎萌新、大佬多多指导！***

## 获取数据 creat_data
### 利用百度AI打标签：baidu.py
API和SDK两种方式。<br>
``` python
# creat_label(texts,
#             interface='SDK',
#             APP_ID=APP_ID,
#             API_KEY=API_KEY,
#             SECRET_KEY=SECRET_KEY)


# texts: 需要打标签的文档列表
# interface: 接口方式，SDK和API
# APP_ID: 百度ai账号信息，默认调用配置文件id_1
# API_KEY: 百度ai账号信息，默认调用配置文件id_1
# SECRET_KEY: 百度ai账号信息，默认调用配置文件id_1
# return: 打好标签的列表，包括原始文档、标签、置信水平、正负面概率、是否错误

from creat_data.baidu import creat_label
import pandas as pd
import numpy as np

results = creat_label(texts=['价格便宜啦，比原来优惠多了',
                             '壁挂效果差，果然一分价钱一分货',
                             '东西一般般，诶呀',
                             '快递非常快，电视很惊艳，非常喜欢',
                             '到货很快，师傅很热情专业。',
                             '讨厌你',
                             '一般'
                             ],
                      interface='SDK')
results = pd.DataFrame(results, columns=['evaluation',
                                         'label',
                                         'confidence',
                                         'positive_prob',
                                         'negative_prob',
                                         'ret',
                                         'msg'])
results['label'] = np.where(results['label'] == 2,
                            '正面',
                            np.where(results['label'] == 1, '中性', '负面'))
print(results)

```
![baidu](https://github.com/renjunxiang/Text-Classification/blob/master/picture/baidu.png)

### 利用阿里云打标签：ali.py
API方式。官方仅给了py2.7版本，和py3.5出入很大，因此重写。<br>
``` python
# creat_label(texts,
#             org_code=org_code,
#             akID=akID,
#             akSecret=akSecret)

# texts: 需要打标签的文档列表
# org_code: 阿里云账号信息，默认调用配置文件id_1
# akID: 阿里云账号信息，默认调用配置文件id_1
# akSecret: 阿里云账号信息，默认调用配置文件id_1
# return: 打好标签的列表，包括原始文档、标签、是否错误

from creat_data.ali import creat_label
import pandas as pd
import numpy as np

results = creat_label(texts=['价格便宜啦，比原来优惠多了',
                             '壁挂效果差，果然一分价钱一分货',
                             '东西一般般，诶呀',
                             '快递非常快，电视很惊艳，非常喜欢',
                             '到货很快，师傅很热情专业。',
                             '讨厌你',
                             '一般'
                             ])
results = pd.DataFrame(results, columns=['evaluation',
                                         'label',
                                         'ret',
                                         'msg'])
results['label'] = np.where(results['label'] == '1', '正面',
                            np.where(results['label'] == '0', '中性',
                                     np.where(results['label'] == '-1', '负面', '非法')))
print(results)

```
![ali](https://github.com/renjunxiang/Text-Classification/blob/master/picture/ali.png)

### 利用腾讯AI打标签：tencent.py
API方式。<br>
``` python
# creat_label(texts,
#             AppID=AppID,
#             AppKey=AppKey)


# texts: 需要打标签的文档列表
# AppID: 腾讯ai账号信息，默认调用配置文件id_1
# AppKey: 腾讯ai账号信息，默认调用配置文件id_1
# return: 打好标签的列表，包括原始文档、标签、置信水平、正负面概率、是否错误

from creat_data.tencent import creat_label
import pandas as pd
import numpy as np

results = creat_label(texts=['价格便宜啦，比原来优惠多了',
                             '壁挂效果差，果然一分价钱一分货',
                             '东西一般般，诶呀',
                             '快递非常快，电视很惊艳，非常喜欢',
                             '到货很快，师傅很热情专业。',
                             '讨厌你',
                             '一般'
                             ])
results = pd.DataFrame(results, columns=['evaluation',
                                         'label',
                                         'confidence',
                                         'ret',
                                         'msg'])
results['label'] = np.where(results['label'] == 1, '正面',
                            np.where(results['label'] == 0, '中性', '负面'))
print(results)

```
![tencent](https://github.com/renjunxiang/Text-Classification/blob/master/picture/tencent.png)

## 文本预处理 sentence_transform
### 文本转tokenizer编码：sentence_2_tokenizer.py
先用jieba分词，再调用keras.preprocessing.text import Tokenizer转编码。<br>
``` python
# sentence_2_tokenizer(train_data,
#                      test_data=None,
#                      num_words=None,
#                      word_index=False)


# train_data: 训练集
# test_data: 测试集
# num_words: 词库大小,None则依据样本自动判定
# word_index: 是否需要索引
# return:返回编码后数组

from sentence_transform.sentence_2_tokenizer import sentence_2_tokenizer

train_data = ['全面从严治党',
              '国际公约和国际法',
              '中国航天科技集团有限公司']
test_data = ['全面从严测试']
train_data_vec, test_data_vec, word_index = sentence_2_tokenizer(train_data=train_data,
                                                                 test_data=test_data,
                                                                 num_words=None,
                                                                 word_index=True)
```
![sentence_2_tokenizer](https://github.com/renjunxiang/Text-Classification/blob/master/picture/sentence_2_tokenizer.png)

### 文本转稀疏矩阵：sentence_2_sparse.py
先用jieba分词，再提供两种稀疏矩阵转换方式：1.转one-hot形式的矩阵，使用pandas的value_counts计数后转dataframe；2.sklearn.feature_extraction.text转成哈希表结构的矩阵。<br>
``` python
# sentence_2_sparse(train_data,
#                   test_data=None,
#                   language='Chinese',
#                   hash=True,
#                   hashmodel='CountVectorizer')

# train_data: 训练集
# test_data: 测试集
# language: 语种
# hash: 是否转哈希存储
# hashmodel: 哈希计数的方式
# return: 返回编码后稀疏矩阵

from sentence_transform.sentence_2_sparse import sentence_2_sparse

train_data = ['全面从严治党',
              '国际公约和国际法',
              '中国航天科技集团有限公司']
test_data = ['全面从严测试']
m, n = sentence_2_sparse(train_data=train_data, test_data=test_data, hash=True)
```
![ex1](https://github.com/renjunxiang/Text-Classification/blob/master/picture/sentence_2_sparse.png)

### 文本转词向量：sentence_2_vec.py
先用jieba分词，再调用gensim.models的word2vec计算词向量。<br>
``` python
# sentence_2_vec(train_data,
#                test_data=None,
#                size=5,
#                window=5,
#                min_count=1)

# train_data: 训练集
# test_data: 测试集
# size: 词向量维数
# window: word2vec滑窗大小
# min_count: word2vec滑窗内词语数量
# return: 返回词向量数组

from sentence_transform.sentence_2_vec import sentence_2_vec

train_data = ['全面从严治党',
              '国际公约和国际法',
              '中国航天科技集团有限公司']
test_data = ['全面从严测试']
train_data, test_data = sentence_2_vec(train_data=train_data,
                                       test_data=test_data,
                                       size=5,
                                       window=5,
                                       min_count=1)
```
![ex2](https://github.com/renjunxiang/Text-Classification/blob/master/picture/sentence_2_vec.png)

## 模型训练 models
### 监督机器学习：sklearn_supervised.py
利用sentence_transform.py文本转稀疏矩阵后，通过sklearn.feature_extraction.text模块转为哈希格式减小存储开销，然后通过常用的机器学习分类模型如SVM和KNN进行学习和预测。本质为将文本转为稀疏矩阵作为训练集的数据，结合标签进行监督学习。<br>
``` python
# sklearn_supervised(language='English',
#                    model_exist=False,
#                    model_path=None,
#                    model_name='SVM',
#                    vector=True,
#                    hashmodel='CountVectorizer',
#                    savemodel=False,
#                    train_dataset=None,
#                    test_data=None)



# language: 语种,中文将调jieba先分词
# model_exist: 模型是否存在
# model_path: 模型路径
# model_name: 机器学习分类模型,SVM,KNN,Logistic
# hashmodel: 哈希方式:CountVectorizer,TfidfTransformer,HashingVectorizer
# savemodel: 保存模型
# train_dataset: 训练集[[数据],[标签]]
# test_data: 测试集[数据]
# return: 预测结果的数组

import numpy as np
import pandas as pd
from models.sklearn_supervised import sklearn_supervised

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

```
![ex3](https://github.com/renjunxiang/Text-Classification/blob/master/picture/文本分类.png)

### 常用层封装：neural_bulit.py
keras的LSTM、Dense、Conv1D封装。<br>
``` python
# neural_bulit(net_shape,
#              optimizer_name='Adagrad',
#              lr=0.001,
#              loss='categorical_crossentropy')


# net_shape: 神经网络格式
# optimizer_name: 优化器
# lr: 学习率
# loss: 损失函数
# return: 返回神经网络模型

from models.neural_bulit import neural_bulit

net_shape = [{'name': 'InputLayer',
              'input_shape': [10, 5],
              },
             {'name': 'Conv1D'
              },
             {'name': 'MaxPooling1D'
              },
             {'name': 'Flatten'
              },
             {'name': 'Dense'
              },
             {'name': 'Dropout'
              },
             {'name': 'softmax'
              }
             ]
model = neural_bulit(net_shape=net_shape,
                     optimizer_name='Adagrad',
                     lr=0.001,
                     loss='categorical_crossentropy')
model.summary()

```
![neural_bulit](https://github.com/renjunxiang/Text-Classification/blob/master/picture/neural_bulit.png)

### LSTM：neural_LSTM.py
keras的LSTM简单封装。<br>
``` python
# neural_LSTM(input_shape,
#             net_shape=[64, 64, 128, 2],
#             optimizer_name='Adagrad',
#             lr=0.001)

# input_shape: 样本数据格式
# net_shape: 神经网络格式
# optimizer_name: 优化器
# lr: 学习率
# return: 返回神经网络模型

from models.neural_LSTM import neural_LSTM

model = neural_LSTM(input_shape=[10, 5],
                    net_shape=[64, 64, 128, 2],
                    optimizer_name='SGD',
                    lr=0.001)
model.summary()
```
![neural_LSTM](https://github.com/renjunxiang/Text-Classification/blob/master/picture/neural_LSTM.png)

### Conv1D：neural_Conv1D.py
keras的Conv1D简单封装。<br>
``` python
# neural_Conv1D(input_shape,
#               net_conv_num=[64, 64],
#               kernel_size=[5, 5],
#               pooling=True,
#               pooling_size=[5, 5],
#               net_dense_shape=[128, 64, 2],
#               optimizer_name='Adagrad',
#               lr=0.001)


# input_shape: 样本数据格式
# net_conv_num: 卷积核数量
# kernel_size: 卷积核尺寸
# pooling_size: 池化尺寸
# pooling: 是否需要池化
# net_dense_shape: 全连接数量
# optimizer_name: 优化器
# lr: 学习率
# return: 返回神经网络模型

from models.neural_Conv1D import neural_Conv1D

model = neural_Conv1D(input_shape=[10, 5],
                      net_conv_num=[64, 64],
                      kernel_size=[5, 5],
                      net_dense_shape=[128, 64, 2],
                      optimizer_name='Adagrad',
                      lr=0.001)
model.summary()
```
![neural_Conv1D](https://github.com/renjunxiang/Text-Classification/blob/master/picture/neural_Conv1D.png)

### 非监督学习：LDA.py
``` python
# LDA(dataset=None,
#     topic_num=5,
#     alpha=0.0002,
#     beta=0.02,
#     steps=500,
#     error=0.1)

# dataset = 数据集,
# topic_num = 主题数,
# alpha = 学习率,
# beta = 正则系数,
# steps = 迭代上限,
# error = 误差阈值

from models.LDA import LDA

dataset = [['document' + str(i) for i in range(1, 11)],
           ['全面从严治党，是十九大报告的重要内容之一。十九大闭幕不久，习近平总书记在十九届中央纪委二次全会上发表重要讲话',
            '根据国际公约和国际法，对沉船进行打捞也要听取船东的意见。打捞工作也面临着很大的风险和困难，如残留凝析油可能再次燃爆',
            '下午召开的北京市第十四届人大常委会第四十四次会议决定任命殷勇为北京市副市长',
            '由中国航天科技集团有限公司所属中国运载火箭技术研究院抓总研制的长征十一号固体运载火箭“一箭六星”发射任务圆满成功',
            '直到2016年7月份，谢某以性格不合为由，向卢女士提出分手，并要求喝分手酒，可谁知，这醉翁之意不在酒哪',
            '湖北男子吴锐在其居住的湖南长沙犯下了一桩大案：跟踪一名开玛莎拉蒂女子',
            '甚而至于得罪了名人或名教授',
            '判决书显示，现年不到30岁的吴锐出生于湖北省天门市，住湖南省长沙县',
            '张某报警后，公安机关在侯某家门前将李某抢劫来的车辆前后别住。李某见状开始倒车',
            '被打女童来自哪里？打人者是谁？1月17日晚，澎湃新闻联系上女童曾某的母亲']]
model = LDA(dataset=dataset, steps=200)
document_topic, topic_word = model.document_recommend_topic(num_topic=2, num_word=8)
print('document_recommend_topic\n', document_topic)
print('topic_recommend_word\n', topic_word)
```
利用sentence_transform.py文本转稀疏矩阵后，对稀疏矩阵进行ALS分解，转为文本-主题矩阵*主题-词语矩阵。<br>
![ex4](https://github.com/renjunxiang/Text-Classification/blob/master/picture/文本主题分类数据.png)
![ex5](https://github.com/renjunxiang/Text-Classification/blob/master/picture/文本主题分类.png)

## 案例 demo
### 机器监督学习的范例：demo_score.py
读取数据集data\demo_score\data.xlsx（商业数据暂时保密，仅提供部分预测结果约1400条），拆分数据为训练集和测试集，通过supervised_classify.py进行机器学习，再对每条文本打分。<br>
训练数据已更新,准确率最高84%<br>
![ex6](https://github.com/renjunxiang/Text-Classification/blob/master/picture/demo_score_1.png)
图片为不同数据处理和不同模型的准确率<br>
![ex7](https://github.com/renjunxiang/Text-Classification/blob/master/picture/demo_score_2.png)

### 机器监督学习+打标签的范例：demo_topic_score.py
读取数据集NLP\data\，关键词：keyword.json，训练集train_data.json<br>，名称的配置文件config.py。然后通过supervised_classify.py对每个主题进行机器学习，再对每条文本打分。<br>
因为没有数据，我自己随便造了几句，训练效果马马虎虎~
![ex8](https://github.com/renjunxiang/Text-Classification/blob/master/picture/文本分类+打标签.png)

### LSTM的范例：demo_LSTM.py
读取数据集data\demo_score\data.xlsx，通过neural_LSTM.py构建LSTM网络并训练（epoch=10），调用keras_log_plot.py可视化训练过程。<br>
![demo_LSTM](https://github.com/renjunxiang/Text-Classification/blob/master/picture/demo_LSTM.png)

### Conv1D的范例：demo_Conv1D.py
读取数据集data\demo_score\data.xlsx，通过neural_Conv1D.py构建Conv1D网络并训练（epoch=10），调用keras_log_plot.py可视化训练过程。<br>
![demo_LSTM](https://github.com/renjunxiang/Text-Classification/blob/master/picture/demo_Conv1D.png)




