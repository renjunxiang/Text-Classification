from keras.models import Sequential
from keras.layers.core import Dense, initializers, Flatten, Dropout,Masking
from keras.layers import Conv1D, InputLayer
from keras.layers.recurrent import LSTM
from keras.layers.pooling import MaxPooling1D
from models.parameter.optimizers import optimizers


net_shape = [{'name': 'InputLayer',
              'input_shape': [10, 5],
              },
             {'name': 'Dropout',
              'rate': 0.2,
              },
             {'name': 'Masking'
              },
             {'name': 'LSTM',
              'units': 16,
              'activation': 'tanh',
              'recurrent_activation': 'hard_sigmoid',
              'dropout': 0.,
              'recurrent_dropout': 0.,
              },
             {'name': 'Conv1D',
              'filters': 64,
              'kernel_size': 3,
              'strides': 1,
              'padding': 'same',
              },
             {'name': 'MaxPooling1D',
              'pool_size': 5,
              'padding': 'same',
              'strides': 2
              },
             {'name': 'Flatten'
              },
             {'name': 'Dense',
              'units': 64
              },
             {'name': 'softmax',
              'units': 2
              }
             ]


def neural_bulit(net_shape,
                 optimizer_name='Adagrad',
                 lr=0.001,
                 loss='categorical_crossentropy'):
    '''
    :param net_shape: 神经网络格式
    :param optimizer_name: 优化器
    :param lr: 学习率
    :param loss: 损失函数
    :param return: 返回神经网络模型
    '''
    model = Sequential()

    for n in range(len(net_shape)):

        if net_shape[n]['name'] == 'InputLayer':
            model.add(InputLayer(input_shape=net_shape[n]['input_shape'],
                                 name='num_' + str(n) + '_InputLayer'))

        elif net_shape[n]['name'] == 'Dropout':
            if 'rate' not in net_shape[n]:
                net_shape[n].update({'rate': 0.2})
            model.add(Dropout(rate=net_shape[n]['rate'],
                              name='num_' + str(n) + '_Dropout'))
        elif net_shape[n]['name'] == 'Masking':
            model.add(Masking(mask_value=0))

        elif net_shape[n]['name'] == 'LSTM':
            if 'units' not in net_shape[n]:
                net_shape[n].update({'units': 16})
            if 'activation' not in net_shape[n]:
                net_shape[n].update({'activation': 'tanh'})
            if 'recurrent_activation' not in net_shape[n]:
                net_shape[n].update({'recurrent_activation': 'hard_sigmoid'})
            if 'dropout' not in net_shape[n]:
                net_shape[n].update({'dropout': 0.})
            if 'recurrent_dropout' not in net_shape[n]:
                net_shape[n].update({'recurrent_dropout': 0.})

            model.add(LSTM(units=net_shape[n]['units'],
                           activation=net_shape[n]['activation'],
                           recurrent_activation=net_shape[n]['recurrent_activation'],
                           implementation=1,
                           dropout=net_shape[n]['dropout'],
                           recurrent_dropout=net_shape[n]['recurrent_dropout'],
                           name='num_' + str(n) + '_LSTM'))

        elif net_shape[n]['name'] == 'Conv1D':
            if 'filters' not in net_shape[n]:
                net_shape[n].update({'filters': 16})
            if 'kernel_size' not in net_shape[n]:
                net_shape[n].update({'kernel_size': 3})
            if 'strides' not in net_shape[n]:
                net_shape[n].update({'strides': 1})
            if 'padding' not in net_shape[n]:
                net_shape[n].update({'padding': 'same'})

            model.add(Conv1D(filters=net_shape[n]['filters'],  # 卷积核数量
                             kernel_size=net_shape[n]['kernel_size'],  # 卷积核尺寸，或者[3]
                             strides=net_shape[n]['strides'],
                             padding=net_shape[n]['padding'],
                             activation='relu',
                             kernel_initializer=initializers.normal(stddev=0.1),
                             bias_initializer=initializers.normal(stddev=0.1),
                             name='num_' + str(n) + '_Conv1D'))

        elif net_shape[n]['name'] == 'MaxPooling1D':
            if 'pool_size' not in net_shape[n]:
                net_shape[n].update({'pool_size': 3})
            if 'strides' not in net_shape[n]:
                net_shape[n].update({'strides': 1})
            if 'padding' not in net_shape[n]:
                net_shape[n].update({'padding': 'same'})
            model.add(MaxPooling1D(pool_size=net_shape[n]['pool_size'],  # 卷积核尺寸，或者[3]
                                   strides=net_shape[n]['strides'],
                                   padding=net_shape[n]['padding'],
                                   name='num_' + str(n) + '_MaxPooling1D'))

        elif net_shape[n]['name'] == 'Flatten':
            model.add(Flatten())

        elif net_shape[n]['name'] == 'Dense':
            if 'units' not in net_shape[n]:
                net_shape[n].update({'units': 16})
            model.add(Dense(units=net_shape[n]['units'],
                            activation='relu',
                            kernel_initializer=initializers.normal(stddev=0.1),
                            name='num_' + str(n) + '_Dense'))

        elif net_shape[n]['name'] == 'softmax':
            if 'units' not in net_shape[n]:
                net_shape[n].update({'units': 16})
            model.add(Dense(units=net_shape[n]['units'],
                            activation='softmax',
                            kernel_initializer=initializers.normal(stddev=0.1),
                            name='num_' + str(n) + '_softmax'))
    optimizer = optimizers(name=optimizer_name, lr=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


if __name__ == '__main__':
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
