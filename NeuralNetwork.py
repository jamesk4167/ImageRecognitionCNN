
import numpy as np
import random


class NeuralNetwork():

    #sizes contains the amount of neurons inside each layer
    def init(self, sizes):
        self.No_Of_layers = len(sizes)
        self.sizes = sizes
        self.weight = [np.random.rand(y,1) for y in sizes[1:]]
        self.bias = [np.random.rand(y,x) for y, x in zip(sizes[:-1], sizes[1:])]

    def sigmoid(z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return sigmoid(z)*(1-sigmoid(z))

    def forward(self, a):
        for bias, weight in (self.bias, self.weight):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self, training_data,epochs, mini_batch_size, eta, test_data= None):

        if test_data: n_test = len(test_data)
        n = len(training_data)

        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0,n,mini_batch_size)
            ]
            for mini_batches in mini_batches:
                self.update_mini_batch(mini_batches, eta)
            if test_data: 
                print ("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} completed".format(i))

    def updateMiniBatches(self, mini_batches, eta):
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weight]

        for x, y in mini_batches:
            delta_nable_b, delta_nabla_w = self.backpropogation(x,y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b,delta_nable_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w,delta_nabla_w)]

            self.weight = [w - (eta/len(mini_batches)) * nw for w, nw in zip(self.weight, nabla_w)]
            self.bias = [b - (eta/len(mini_batches)) * nb for b, nb in zip(self.bias, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)



    












        

        return "Test"


    def backward():
        return "test"