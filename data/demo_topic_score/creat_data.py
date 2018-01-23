import json

keyword = {
    '手机': ['小米手机', '华为手机', '苹果手机', 'iphone'],
    '电商': ['淘宝', '京东', '一号店']
}
with open('D:/github/machine-learning/NLP/data/demo_topic_score/keyword.json',
          mode='a', encoding='utf-8') as f:
    json.dump(keyword, f)

train_dataset = {'手机': {'data':
                            ['小米手机好用',
                             '华为手机不错',
                             '苹果手机牛逼',
                             '小米手机烫手',
                             '华为手机质量差',
                             'iphone太贵'],
                        'label': ['正面',
                                  '正面',
                                  '正面',
                                  '负面',
                                  '负面',
                                  '负面']},
                 '电商': {'data':
                              ['淘宝好用',
                               '京东不错',
                               '一号店便宜',
                               '淘宝假货',
                               '京东二手多',
                               '一号店服务差'],
                          'label': ['正面',
                                    '正面',
                                    '正面',
                                    '负面',
                                    '负面',
                                    '负面']}
                 }

with open('D:/github/machine-learning/NLP/data/demo_topic_score/train_data.json',
          mode='a', encoding='utf-8') as f:
    json.dump(train_dataset, f)
