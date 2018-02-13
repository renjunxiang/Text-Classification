from creat_data.config import tencent
import pandas as pd
import numpy as np
import requests
import json
import time
import random
import hashlib
from urllib import parse
from collections import OrderedDict

AppID = tencent['account']['id_1']['APP_ID']
AppKey = tencent['account']['id_1']['AppKey']

def cal_sign(params_raw,AppKey=AppKey):
    # 官方文档例子为php，给出python版本
    # params_raw = {'app_id': '10000',
    #               'time_stamp': '1493449657',
    #               'nonce_str': '20e3408a79',
    #               'key1': '腾讯AI开放平台',
    #               'key2': '示例仅供参考',
    #               'sign': ''}
    # AppKey = 'a95eceb1ac8c24ee28b70f7dbba912bf'
    # cal_sign(params_raw=params_raw,
    #          AppKey=AppKey)
    # 返回：BE918C28827E0783D1E5F8E6D7C37A61
    params = OrderedDict()
    for i in sorted(params_raw):
        if params_raw[i] != '':
            params[i] = params_raw[i]
    newurl = parse.urlencode(params)
    newurl += ('&app_key=' + AppKey)
    sign = hashlib.md5(newurl.encode("latin1")).hexdigest().upper()
    return sign


def creat_label(texts,
                AppID=AppID,
                AppKey=AppKey):
    '''
    :param texts: 需要打标签的文档列表
    :param AppID: 腾讯ai账号信息，默认调用配置文件id_1
    :param AppKey: 腾讯ai账号信息，默认调用配置文件id_1
    :return: 打好标签的列表，包括原始文档、标签、置信水平、是否成功
    '''

    url = tencent['api']['nlp_textpolar']['url']
    results = []
    # 逐句调用接口判断
    for one_text in texts:
        params = {'app_id': AppID,
                  'time_stamp': int(time.time()),
                  'nonce_str': ''.join([random.choice('1234567890abcdefghijklmnopqrstuvwxyz') for i in range(10)]),
                  'sign': '',
                  'text': one_text}
        params['sign'] = cal_sign(params_raw=params,
                                  AppKey=AppKey)  # 获取sign
        r = requests.post(url=url,
                          params=params)  # 获取分析结果
        result = json.loads(r.text)
        # print(result)
        results.append([one_text,
                        result['data']['polar'],
                        result['data']['confd'],
                        result['ret'],
                        result['msg']
                        ])
    return results


if __name__ == '__main__':
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
