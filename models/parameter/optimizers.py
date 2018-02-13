from keras.optimizers import SGD, Adagrad, Adam


def optimizers(name='SGD', lr=0.001):
    if name == 'SGD':
        optimizers_fun = SGD(lr=lr)
    elif name == 'Adagrad':
        optimizers_fun = Adagrad(lr=lr)
    elif name == 'Adam':
        optimizers_fun = Adam(lr=lr)

    return optimizers_fun
