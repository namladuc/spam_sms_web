import numpy as np
import math
from cvxopt import matrix, solvers

class HardMargin:
    def __init__(self, epsilon = 1e-6):
        self.epsilon = epsilon

    def fit(self, X, y):
        x0 = np.ones(( X.shape[0], 1))
        X_value = np.concatenate((x0, X), axis = 1)
        for i in range(X_value.shape[0]):
            X_value[i,:] = X_value[i,:] * y[i]
        y = np.asarray([y])

        N = X_value.shape[0]
        # build P ~ K
        V = X_value.copy().T
        P = matrix(V.T.dot(V)) # P ~ K in slide see definition of V, K near eq (8)
        q = matrix(-np.ones((N, 1))) # all-one vector

        # build A, b, G, h
        G = matrix(-np.eye(N)) # for all lambda_n >= 0! note that we solve -g(lambda) -> min
        h = matrix(np.zeros((N, 1)))
        A = matrix(y, (1, N), 'd') # the equality constrain is actually y^T lambda = 0
        b = matrix(np.zeros((1, 1)))
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        l = np.array(sol['x']) # lambda

        S = np.where(l > self.epsilon)[0]
        VS = V[:, S]
        XS = X_value.T[:, S]
        yS = y[:, S]
        lS = l[S]

        w = VS.dot(lS)
        w0 = np.mean(yS.T - w.T.dot(XS))

        w0 = np.asarray([[w0]])
        self.w_best = np.concatenate((w0, w), axis = 0)

    # Define h_w(x):= W^T.x + w_0 = \bar{W}^T . \bar{x}
    def h_w_x(self, w, x):
        return np.sign(np.dot(w.T, x))

    def predict(self, x_input):
        """
            x_test must have a one column in front
        """
        x0 = np.ones(( x_input.shape[0], 1))
        X_value = np.concatenate((x0, x_input), axis = 1)
        x_test = np.concatenate((np.ones((1, X_value.shape[0])), X_value.T), axis = 0)
        num_att, num_x = x_test.shape
        y_pred = []
        for index in range(num_x):
            xi = x_test[:, index].reshape(num_att, 1)
            y_hat = self.h_w_x(self.w_best, xi)[0][0]
            y_pred.append(y_hat)
        return np.asarray(y_pred)
    
class SoftMargin:
    def __init__(self, epsilon = 1e-6, C = 100):
        self.epsilon = epsilon
        self.C = C
        
    def fit(self, X, y):
        x0 = np.ones(( X.shape[0], 1))
        X_value = np.concatenate((x0, X), axis = 1)
        for i in range(X_value.shape[0]):
            X_value[i,:] = X_value[i,:] * y[i]
        y = np.asarray([y])

        N = X_value.shape[0]
        V = X_value.copy().T
        # build K
        K = matrix(V.T.dot(V))
        p = matrix(-np.ones((N, 1)))
        # build A, b, G, h 
        G = matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = matrix(np.vstack((np.zeros((N, 1)), self.C*np.ones((N, 1)))))
        A = matrix(y, (1, N), 'd') 
        b = matrix(np.zeros((1, 1))) 
        
        solvers.options['show_progress'] = False
        sol = solvers.qp(K, p, G, h, A, b)
        l = np.array(sol['x']) # lambda

        S = np.where(l > self.epsilon)[0] # support set 
        S2 = np.where(l < .999*self.C)[0] 
        
        M = [val for val in S if val in S2] # intersection of two lists
        XT = X_value.T # we need each column to be one data point in this alg
        VS = V[:, S]
        lS = l[S]
        yM = y[M]
        XM = XT[:, M]

        w = VS.dot(lS).reshape(-1, 1)
        w0 = np.mean(yM.T - w.T.dot(XM))

        if math.isnan(w0):
            w0 = 0

        w0 = np.asarray([[w0]])
        self.w_best = np.concatenate((w0, w), axis = 0)
        
    # Define h_w(x):= W^T.x + w_0 = \bar{W}^T . \bar{x}
    def h_w_x(self, w, x):
        return np.sign(np.dot(w.T, x))

    def predict(self, x_input):
        """
            x_test must have a one column in front
        """
        x0 = np.ones(( x_input.shape[0], 1))
        X_value = np.concatenate((x0, x_input), axis = 1)
        x_test = np.concatenate((np.ones((1, X_value.shape[0])), X_value.T), axis = 0)
        num_att, num_x = x_test.shape
        y_pred = []
        for index in range(num_x):
            xi = x_test[:, index].reshape(num_att, 1)
            y_hat = self.h_w_x(self.w_best, xi)[0][0]
            y_pred.append(y_hat)
        return np.asarray(y_pred)