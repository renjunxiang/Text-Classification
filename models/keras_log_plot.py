import matplotlib.pyplot as plt
def keras_log_plot(train_log=None):
    plt.plot(train_log['acc'],label='acc',color='red')
    plt.plot(train_log['loss'],label='loss',color='yellow')
    plt.plot(train_log['val_acc'],label='val_acc',color='green')
    plt.plot(train_log['val_loss'],label='val_loss',color='blue')
    plt.legend()
    plt.show()
