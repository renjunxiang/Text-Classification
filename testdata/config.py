rawdata = {'positive':
               {'path': 'D:/github/Text-Classification/data/demo_score/data.xlsx',
                'sheetname': 'positive'},
           'negative':
               {'path': 'D:/github/Text-Classification/data/demo_score/data.xlsx',
                'sheetname': 'negative'}
           }
filename = 'D:/github/Text-Classification/testdata/6000document/'

test = {'20180126':
            {'update':
                 {'path': filename + '20180126/update_v1.xlsx',
                  'sheetname': 'Sheet1'},
             'test': {'path': filename + '20180126/test_v1.xlsx',
                      'sheetname': 'Sheet1'}
             },
        '20180130':
            {'update':
                 {'path': filename + '20180130/update_v1.xlsx',
                  'sheetname': 'Sheet1'},
             'test': {'path': filename + '20180130/test_v1.xlsx',
                      'sheetname': 'Sheet1'}
             }
        }
