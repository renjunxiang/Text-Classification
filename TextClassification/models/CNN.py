from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import Conv1D, GlobalMaxPool1D, Dropout


def CNN(input_dim, input_length, vec_size, output_shape, output_type='multiple'):
    '''
    Creat CNN net,use Embedding+CNN1D+GlobalMaxPool1D+Dense.
    You can change filters and dropout rate in code.
    This model was simple and much faster than RNN, but not stable.
    I got low score in Kaggle "Toxic Comment Classification Challenge",
    but top2 in preliminary of "CAIL2018(中国‘法研杯’法律智能挑战赛)"

    :param input_dim: Size of the vocabulary
    :param input_length:Length of input sequences
    :param vec_size:Dimension of the dense embedding
    :param output_shape:Target shape,target should be one-hot term
    :param output_type:last layer type,multiple(activation="sigmoid") or single(activation="softmax")
    :return:keras model
    '''
    data_input = Input(shape=[input_length])
    word_vec = Embedding(input_dim=input_dim + 1,
                         input_length=input_length,
                         output_dim=vec_size,
                         mask_zero=0,
                         name='Embedding')(data_input)
    x = Conv1D(filters=128, kernel_size=[3], strides=1, padding='same', activation='relu')(word_vec)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.1)(x)
    if output_type == 'multiple':
        x = Dense(output_shape, activation='sigmoid')(x)
        model = Model(inputs=data_input, outputs=x)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    elif output_type == 'single':
        x = Dense(output_shape, activation='softmax')(x)
        model = Model(inputs=data_input, outputs=x)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    else:
        raise ValueError('output_type should be multiple or single')
    return model


if __name__ == '__main__':
    model = CNN(input_dim=10, input_length=10, vec_size=10, output_shape=10, output_type='multiple')
    model.summary()
