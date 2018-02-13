from aip import AipNlp
from creat_data.config import baidu
import pandas as pd
import numpy as np
import json
import requests

APP_ID = baidu['account']['id_1']['APP_ID']
API_KEY = baidu['account']['id_1']['API_KEY']
SECRET_KEY = baidu['account']['id_1']['SECRET_KEY']


# 逐句调用接口判断
def creat_label(texts,
                interface='SDK',
                APP_ID=APP_ID,
                API_KEY=API_KEY,
                SECRET_KEY=SECRET_KEY):
    '''
    :param texts: 需要打标签的文档列表
    :param interface: 接口方式，SDK和API
    :param APP_ID: 百度ai账号信息，默认调用配置文件id_1
    :param API_KEY: 百度ai账号信息，默认调用配置文件id_1
    :param SECRET_KEY: 百度ai账号信息，默认调用配置文件id_1
    :return: 打好标签的列表，包括原始文档、标签、置信水平、正负面概率、是否成功
    '''
    # 创建连接
    client = AipNlp(APP_ID=APP_ID,
                    API_KEY=API_KEY,
                    SECRET_KEY=SECRET_KEY)
    results = []
    if interface == 'SDK':
        for one_text in texts:
            result = client.sentimentClassify(one_text)
            if 'error_code' in result:
                results.append([one_text,
                                0,
                                0,
                                0,
                                0,
                                result['error_code'],
                                result['error_msg']
                                ])
            else:
                results.append([one_text,
                                result['items'][0]['sentiment'],
                                result['items'][0]['confidence'],
                                result['items'][0]['positive_prob'],
                                result['items'][0]['negative_prob'],
                                0,
                                'ok'
                                ])
    elif interface == 'API':
        # 获取access_token
        url = baidu['access_token_url']
        params = {'grant_type': 'client_credentials',
                  'client_id': baidu['account']['id_1']['API_KEY'],
                  'client_secret': baidu['account']['id_1']['SECRET_KEY']}
        r = requests.post(url, params=params)
        access_token = json.loads(r.text)['access_token']

        url = baidu['api']['sentiment_classify']['url']
        params = {'access_token': access_token}
        headers = {'Content-Type': baidu['api']['sentiment_classify']['Content-Type']}
        for one_text in texts:
            data = json.dumps({'text': one_text})
            r = requests.post(url=url,
                              params=params,
                              headers=headers,
                              data=data)
            result = json.loads(r.text)
            if 'error_code' in result:
                results.append([one_text,
                                0,
                                0,
                                0,
                                0,
                                result['error_code'],
                                result['error_msg']
                                ])
            else:
                results.append([one_text,
                                result['items'][0]['sentiment'],
                                result['items'][0]['confidence'],
                                result['items'][0]['positive_prob'],
                                result['items'][0]['negative_prob'],
                                0,
                                'ok'
                                ])
    else:
        print('ERROR: No interface named %s' % (interface))
    return results


if __name__ == '__main__':
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
