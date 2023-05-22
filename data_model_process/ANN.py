import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ANN:
    def __init__(self, learning_rate=0.01, layer_dims = (1, 7, 1), n_iter=50, print_cost = False, random_state=1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.layer_dims = layer_dims
        self.print_cost = print_cost

    def fit(self, X, y_true):
        self.costs = []
        self.parameters = self.initialize_parameters(self.layer_dims)
        y = np.asarray([y_true])
        for i in range(self.n_iter):
            AL, caches = self.forward(X.T)

            cost = self.compute_cost(AL, Y=y)

            grads = self.backward(AL=AL, Y=y, caches=caches)

            self.parameters = self.update_parameters(parameters = self.parameters, grads=grads, learning_rate=self.learning_rate)
            
            if self.print_cost and i % 100 == 0 or i == self.n_iter - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == self.n_iter:
                self.costs.append(cost)
      
    def plot_costs(self):
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()
        


    def initialize_parameters(self, layer_dims):
        rgen = np.random.RandomState(self.random_state)
        self.parameters = {}
        L = len(self.layer_dims)
        for l in range(1, L):
            self.parameters['W' + str(l)] = rgen.randn(self.layer_dims[l], self.layer_dims[l-1]) * (2. / np.sqrt(self.layer_dims[l-1]))
            self.parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

            assert(self.parameters['W' + str(l)].shape == (self.layer_dims[l], layer_dims[l - 1]))
            assert(self.parameters['b' + str(l)].shape == (self.layer_dims[l], 1))
        
        return self.parameters
    
    def forward(self, X):
        caches = []
        A = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_function(A_prev, self.parameters['W' + str(l)],
                                                       self.parameters['b' + str(l)],
                                                       activation='relu')
            caches.append(cache)
        
        AL, cache = self.linear_activation_function(A, self.parameters['W' + str(L)],
                                                    self.parameters['b' + str(L)],
                                                    activation='sigmoid')
        caches.append(cache)
        assert(AL.shape == (1,X.shape[1]))

        return AL, caches
    
    def backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[L-1]
        grads['dA' + str(L- 1)], grads['dW' + str(L)], grads['db' + str(L)] = self.linear_activation_backward(dAL, current_cache, activation='sigmoid')

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads['dA' + str(l+1)],
                                                                             cache=current_cache,
                                                                             activation='relu')
            grads['dA' + str(l)] = dA_prev_temp
            grads['dW' + str(l+1)] = dW_temp
            grads['db' + str(l+1)] = db_temp

        return grads

    def predict(self, X, y):
        """
        This function is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """
        
        m = X.shape[1]
        n = len(self.parameters) // 2 # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas, caches = self.forward(X)

        
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        print("Accuracy: "  + str(np.sum((p == y)/m)))
            
        return p

    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
                      parameters["W" + str(l)] = ... 
                      parameters["b" + str(l)] = ...
        """
        
        L = len(parameters) // 2 # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
            
        return parameters

    
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

        cost = np.squeeze(cost)
        assert(cost.shape == ())

        return cost

    
    def linear_forward(self, A, W, b):
        Z = W.dot(A) + b

        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return (Z, cache)
    
    def linear_activation_function(self, A_prev, W, b, activation):
        if activation == 'sigmoid':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
        elif activation == 'relu':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
        else:
            print('\033[91mError! Please make sure you have passed the value correctly in the \"activation\" parameter')
        
        assert(A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
        return A, cache
    
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1. / m) * np.dot(dZ, A_prev.T)
        db = (1. / m) * np.sum(dZ, axis = 1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert(dA_prev.shape == A_prev.shape)
        assert(dW.shape == W.shape)
        assert(db.shape == b.shape)


        return dA_prev, dW, db
    
    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache

        if activation == 'relu':
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        elif activation == 'sigmoid':
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        else:
            print('\033[91mError! Please make sure you have passed the value correctly in the \"activation\" parameter')

        return dA_prev, dW, db


    def relu_backward(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy= True)
        dZ[Z <= 0] = 0
        assert(dZ.shape == Z.shape)
        return dZ
    
    def sigmoid_backward(self, dA, cache):
        Z = cache
        s = 1. / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        assert(dZ.shape == Z.shape)

        return dZ

    def relu(self, Z):
        A = np.maximum(0,Z)
        
        assert(A.shape == Z.shape)
        
        cache = Z 
        return A, cache
    
    def sigmoid(self, Z):
        """
        Implements the sigmoid activation in numpy
        
        Arguments:
        Z -- numpy array of any shape
        
        Returns:
        A -- output of sigmoid(z), same shape as Z
        cache -- returns Z as well, useful during backpropagation
        """
        
        A = 1/(1+np.exp(-Z))
        cache = Z
        
        return A, cache