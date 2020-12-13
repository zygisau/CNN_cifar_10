from tensorflow.keras.optimizers import SGD


def load_P1_options():
    P1 = {
        'l_rate': 0.001,
        'momentum': 0.9,
        'n_epoch': 100,
        'batch_size': 64,
        'verbose': 2,
        'optimizer': SGD,
        'loss_func': 'categorical_crossentropy'
    }
    return P1.values()


def load_P2_options():
    P2 = {
        'l_rate': 0.05,
        'momentum': 0.9,
        'n_epoch': 100,
        'batch_size': 64,
        'verbose': 2,
        'optimizer': SGD,
        'loss_func': 'categorical_crossentropy'
    }
    return P2.values()


def load_P3_options():
    P3 = {
        'l_rate': 0.001,
        'momentum': 0.8,
        'n_epoch': 150,
        'batch_size': 128,
        'verbose': 2,
        'optimizer': SGD,
        'loss_func': 'categorical_crossentropy'
    }
    return P3.values()
