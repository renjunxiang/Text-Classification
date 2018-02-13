from keras.models import Sequential
from keras.layers.core import Masking, Dense, initializers
from keras.layers.recurrent import LSTM
from models.parameter.optimizers import optimizers

def neural_LSTM(input_shape,
                net_shape=[64,64,128,2],
                optimizer_name='Adagrad',
                lr=0.001):
    '''
    
    :param input_shape: 样本数据格式
    :param net_shape: 神经网络格式
    :param optimizer_name: 优化器
    :param lr: 学习率
    :param return: 返回神经网络模型
    '''
    model = Sequential()
    # 识别之前的'截断/填充',跳过填充
    model.add(Masking(mask_value=0, input_shape=input_shape))
    #增加LSTM层
    model.add(LSTM(units=net_shape[0],
                   activation='tanh',
                   recurrent_activation='hard_sigmoid',
                   implementation=1,
                   dropout=0.2,
                   kernel_initializer=initializers.normal(stddev=0.1),
                   name='LSTM'))
    #增加全连接隐藏层
    for n,units in enumerate(net_shape[1:-1]):
        model.add(Dense(units=units,
                        activation='relu',
                        kernel_initializer=initializers.normal(stddev=0.1),
                        name='Dense'+str(n)))
    #增加最后的softmax层
    model.add(Dense(units=net_shape[-1],
                    activation='softmax',
                    kernel_initializer=initializers.normal(stddev=0.1),
                    name='softmax'))

    optimizer = optimizers(name=optimizer_name, lr=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = neural_LSTM(input_shape=[10, 5],
                        net_shape=[64, 64, 128, 2],
                        optimizer_name='SGD',
                        lr=0.001)
    model.summary()
