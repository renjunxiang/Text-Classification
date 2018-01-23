import numpy as np
import pandas as pd
from gensim.models import word2vec
import jieba

jieba.setLogLevel('WARN')

train_data = ['全面从严治党', '国际公约和国际法', '中国航天科技集团有限公司', '全面从严测试']
train_data = [[word for word in jieba.lcut(sample) if word != ' ']  for sample in train_data]



# sentences=word2vec.Text8Corpus(train_data)
model=word2vec.Word2Vec(train_data, size=5, window=5, min_count=1)
model.similarity(w1='国王',w2='喜欢')
model['国王']
train_data = [[model[word] for word in sample] for sample in train_data]


