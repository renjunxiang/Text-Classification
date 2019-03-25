import json
import pandas as pd
import os

localpath = os.path.dirname(__file__)


def load_data(data_type='single'):
    if data_type == 'single':
        data = pd.read_csv(localpath + "/data/data_single.csv", encoding='utf8')
    elif data_type == 'multiple':
        with open(localpath + '/data/data_multiple.json', mode='r', encoding='utf8') as f:
            data_raw = f.readlines()
        data = [json.loads(i) for i in data_raw]
    return data
