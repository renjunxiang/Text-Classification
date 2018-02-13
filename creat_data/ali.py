from creat_data.config import ali
import datetime
import hashlib
import base64
from urllib.parse import urlparse
import hmac
import pandas as pd
import numpy as np
import requests
import json
import time

org_code = ali['account']['id_1']['org_code']
akID = ali['account']['id_1']['akID']
akSecret = ali['account']['id_1']['akSecret']

def creat_label(texts,
                org_code=org_code,
                akID=akID,
                akSecret=akSecret):
    '''
    :param texts: 需要打标签的文档列表
    :param AppID: 腾讯ai账号信息，默认调用配置文件id_1
    :param AppKey: 腾讯ai账号信息，默认调用配置文件id_1
    :return: 打好标签的列表，包括原始文档、标签、置信水平、是否成功
    '''
    url = org_code.join(ali['api']['Sentiment']['url'].split('{org_code}'))

    results = []

    def to_sha1_base64(stringToSign, akSecret):
        hmacsha1 = hmac.new(akSecret.encode('utf-8'),
                            stringToSign.encode('utf-8'),
                            hashlib.sha1)
        return base64.b64encode(hmacsha1.digest()).decode('utf-8')

    # 逐句调用接口判断
    for one_text in texts:
        # one_text = '喜欢'
        time_now = datetime.datetime.strftime(datetime.datetime.utcnow(), "%a, %d %b %Y %H:%M:%S GMT")
        # time_now = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.localtime()) #这个也可以
        options = {'url': url,
                   'method': 'POST',
                   'headers': {'accept': 'application/json',
                               'content-type': 'application/json',
                               'date': time_now,
                               'authorization': ''},
                   'body': json.dumps({'text': one_text}, separators=(',', ':'))}

        body = ''
        if 'body' in options:
            body = options['body']
        # print(body)
        bodymd5 = ''
        if not body == '':
            bodymd5 = base64.b64encode(
                hashlib.md5(json.dumps({'text': one_text}, separators=(',', ':')).encode('utf-8')).digest()).decode(
                'utf-8')

        # print(bodymd5)

        urlPath = urlparse(url)
        if urlPath.query != '':
            urlPath = urlPath.path + "?" + urlPath.query
        else:
            urlPath = urlPath.path
        stringToSign = 'POST' + '\n' + \
                       options['headers']['accept'] + '\n' + \
                       bodymd5 + '\n' + \
                       options['headers']['content-type'] + '\n' \
                       + options['headers']['date'] + '\n' + urlPath

        # print(stringToSign)
        signature = to_sha1_base64(stringToSign=stringToSign,
                                   akSecret=akSecret)
        # print(signature)
        authHeader = 'Dataplus ' + akID + ':' + signature
        # print(authHeader)
        options['headers']['authorization'] = authHeader
        r = requests.post(url=url,
                          headers={'accept': 'application/json',
                                   'content-type': 'application/json',
                                   'date': time_now,
                                   'authorization': authHeader},
                          data=json.dumps({'text': one_text}, separators=(',', ':')))  # 获取分析结果
        try:
            result = json.loads(r.text)
            # print(result)
            results.append([one_text,
                            result['data']['text_polarity'],
                            0,
                            'ok'
                            ])
        except:
            results.append([one_text,
                            -100,
                            -100,
                            'error'
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
                                             'ret',
                                             'msg'])
    results['label'] = np.where(results['label'] == '1', '正面',
                                np.where(results['label'] == '0', '中性',
                                         np.where(results['label'] == '-1', '负面', '非法')))
    print(results)


