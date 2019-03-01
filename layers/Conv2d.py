import numpy as np


class conv2d:
    # need to initialise the conv2d layer(using same params used in tensorflowCNN)
    def __init__(self,filters,kernel_size,input_shape,strides, padding):
        

        self.filters = filters
        self.kernel_size = kernel_size
        self.Height, self.width, self.channels = input_shape

        self.strides = strides
        self.padding = padding
        #set weights randomly, 
        self.weights = np.random.rand(filters + self.Height, self.width, self.channels) / np.sqrt(filters)

        #need to set weights out

        #need to set up forward pass and backpass
    def forward():
        return "test"


    def backward():
        return "test"