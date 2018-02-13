baidu = {'access_token_url': 'https://aip.baidubce.com/oauth/2.0/token',
         'api': {
             'sentiment_classify': {
                 'Content-Type': 'application/json',
                 'url': 'https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify'}},
         'account': {
             'id_1': {'APP_ID': '000',
                      'API_KEY': '000',
                      'SECRET_KEY': '000'},
             'id_2': {'APP_ID': '000',
                      'API_KEY': '000',
                      'SECRET_KEY': '000'}}
         }

tencent = {'api': {
    'nlp_textpolar': {
        'url': 'https://api.ai.qq.com/fcgi-bin/nlp/nlp_textpolar'}},
    'account': {
        'id_1': {'APP_ID': '000',
                 'AppKey': '000'},
        'id_2': {'APP_ID': '000',
                 'API_KEY': '000'}}
}

ali = {'api': {
    'Sentiment': {
        'url': 'https://dtplus-cn-shanghai.data.aliyuncs.com/{org_code}/nlp/api/Sentiment/ecommerce'}},
    'account': {
        'id_1': {'org_code': '000',
                 'akID': '000',
                 'akSecret': '000'
                 },
        'id_2': {'org_code': '000',
                 'akID': '000',
                 'akSecret': '000'
                 }}
}
