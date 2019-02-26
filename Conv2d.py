import numpy as np


class conv2d:

    def __init__(self,filters,kernel_size,input_shape,strides, padding):


        self.filters = filters
        self.kernel_size = kernel_size
        self.Height, self.width, self.channels = input_shape

        self.strides = strides
        self.padding = padding

        self.weights = np.random.rand(filters + self.Height, self.width, self.channels) / np.sqrt(filters)
