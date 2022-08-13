import numpy as np
from sklearn.preprocessing import MinMaxScaler

class NN:
    def __init__(self, **kwargs):
        self.hidden_nodes = kwargs["hn"]
        self.bias_in_bound = kwargs["bib"]
        self.bias_hidden_bound = kwargs["bhb"]
        self.w_bound = kwargs["wb"]
        self.k_bound = kwargs["kb"]
        self.tol = kwargs["tol"]
        self.learning_rate = kwargs["lr"]
        self.max_epochs = kwargs["me"]
        
        self.Xscaler, self.yscaler = MinMaxScaler(kwargs["sri"]), MinMaxScaler(kwargs["sro"])

        self.W, self.What, self.K, self.Khat = None, None, None, None
        self.Bi, self.Bh = None, None
        
        self.loss = []

    def fit(self, X, y):
        X = self.Xscaler.fit_transform(X)
        Xhat = np.hstack((X, np.ones((X.shape[0], 1))))
        y = self.yscaler.fit_transform(y)
        
        self.Bi = np.random.uniform(0, self.bias_in_bound, size=(1, self.hidden_nodes))
        self.Bh = np.random.uniform(0, self.bias_hidden_bound, size=(1, y.shape[1]))

        self.W = np.random.uniform(0, self.w_bound, size=(Xhat.shape[1]-1, self.hidden_nodes))
        self.What = np.vstack((self.W, self.Bi))
        self.K = np.random.uniform(0, self.k_bound, size=(self.hidden_nodes, y.shape[1]))
        self.Khat = np.vstack((self.K, self.Bh))

        SSEold = 100
        epoch = 0
        flag = True

        while flag and epoch < self.max_epochs:
            epoch += 1

            Vstar = Xhat @ self.What
            V = self.__activate(Vstar)
            Vhat = np.hstack((V, np.ones((V.shape[0], 1))))

            O = Vhat @ self.Khat

            e = y - O

            dW = Xhat.T @ (((e * 1) @ self.Khat.T) * self.__deactivate(Vhat))
            dWstar = dW[:, :-1]
            self.What += 2 * self.learning_rate(epoch) * dWstar

            dK = Vhat.T @ (e * 1)
            self.Khat += 2 * self.learning_rate(epoch) * dK

            SSEnew = np.sum(e**2)
            self.loss.append(SSEnew)
            if np.abs(SSEnew - SSEold) < self.tol:
                flag = False
            else:
                SSEold = SSEnew
        
    def predict(self, X):
        X = self.Xscaler.transform(X)
        Xhat = np.hstack((X, np.ones((X.shape[0], 1))))

        Vstar = Xhat @ self.What
        V = self.__activate(Vstar)
        Vhat = np.hstack((V, np.ones((V.shape[0], 1))))

        O = Vhat @ self.Khat

        return self.yscaler.inverse_transform(O)

    @staticmethod
    def __activate(u):
        return 1 / (1 + np.exp(-4 * u))

    @staticmethod
    def __deactivate(u):
        return 4 * u * (1 - u)