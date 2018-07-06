from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import GRU, Bidirectional, GlobalMaxPool1D, Dropout


def RNN(input_dim, input_length, vec_size, output_shape, output_type='multiple'):
    '''
    Creat RNN net,use Embedding+GRU+GlobalMaxPool1D+Dense.
    You can change filters and dropout rate.
    It was simple but effective and used in several competitions and projects.

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
    x = Bidirectional(GRU(50, return_sequences=True))(word_vec)
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
    model = RNN(input_dim=10, input_length=10, vec_size=10, output_shape=10, output_type='multiple')
    model.summary()
