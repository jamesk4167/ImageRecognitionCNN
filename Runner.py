import numpy as np

from layers.LoadImage import *
from layers.layers import *



def make_mnist_cnn(X_dim, num_class):
    conv = Conv(X_dim, n_filter=32, h_filter=3,
                w_filter=3, stride=1, padding=1)
    relu_conv = ReLU()
    maxpool = Maxpool(conv.out_dim, size=2, stride=1)
    flat = Flatten()
    fc = FullyConnected(np.prod(maxpool.out_dim), num_class)
    return [conv, relu_conv, maxpool, flat, fc]


def make_fruit_cnn(X_dim, num_class):
    conv = Conv(X_dim, n_filter=16, h_filter=5,
                w_filter=5, stride=1, padding=2)
    relu = ReLU()
    maxpool = Maxpool(conv.out_dim, size=2, stride=2)
    conv2 = Conv(maxpool.out_dim, n_filter=20, h_filter=5,
                 w_filter=5, stride=1, padding=2)
    relu2 = ReLU()
    maxpool2 = Maxpool(conv2.out_dim, size=2, stride=2)
    flat = Flatten()
    fc = FullyConnected(np.prod(maxpool2.out_dim), num_class)
    return [conv, relu, maxpool, conv2, relu2, maxpool2, flat, fc]


if __name__ == "__main__":
    X, y = loadTrainingData()
    
    
    fruit_dims = (3, 64, 64)
    cnn = CNN(make_fruit_cnn(fruit_dims, num_class=20))
    cnn = sgd(cnn, X, y, minibatch_size=10, epoch=200,
                        learning_rate=0.01, X_test=X_test, y_test=y_test)