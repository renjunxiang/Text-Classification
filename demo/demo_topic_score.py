import json
import numpy as np
import pandas as pd
from models.supervised_classify import supervised_classify
from data.demo_topic_score.config import word_name

with open('D:/github/Text-Classification/data/demo_topic_score/keyword.json', encoding='utf-8') as f:
    keyword = json.load(f)
with open('D:/github/Text-Classification/data/demo_topic_score/train_data.json', encoding='utf-8') as f:
    train_data = json.load(f)

test_data=['京东华为手机牛逼','淘宝假货','小米手机不错','淘宝iphone假货']
topics=keyword.keys()
result=[{word_name['document']:one_test,
         word_name['topics']:[]} for one_test in test_data]
for topic in topics:
    train_dataset=[train_data[topic]['data'],train_data[topic]['label']]
    topic_keyword=keyword[topic]
    predict = supervised_classify(train_dataset=train_dataset,
                                 test_data=test_data,
                                 model_name='SVM',
                                 language='Chinese')
    for n,one_test in enumerate(test_data):
        if any([topic_keyword_one in one_test for topic_keyword_one in topic_keyword]):
            result[n][word_name['topics']].append({word_name['topic']:topic,
                                                   word_name['score']:predict[n]})
result_table=pd.DataFrame({word_name['document']:[i[word_name['document']] for i in result],
                           word_name['topics']:[i[word_name['topics']] for i in result]},
                          columns=[word_name['document'],word_name['topics']])
# print(result_table)
result_table.to_excel('D:/github/Text-Classification/data/demo_topic_score/result.xlsx',index=False)

