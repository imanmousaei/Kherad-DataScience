import numpy as np


class LinearRegression:

    def __init__(self, init_theta=None, alpha=0.01, n_iter=100):
        self.alpha = alpha  # learning rate
        self.n_iter = n_iter  # number of iterations to hepefully converge
        self.theta = init_theta
        self.JHist = None

    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        '''
        # n,d = X.shape
        n = len(y)
        self.JHist = []
        for i in range(self.n_iter):
            prediction = self.predict(X, theta)
            theta = theta -(1/n)*self.alpha*( X.T.dot((prediction - y)))

            self.JHist.append((self.computeCost(X, y, theta), theta))
            print("Iteration: ", i+1, " Cost: ", self.JHist[i][0], " Theta: ", theta)
            
        return theta

    def computeCost(self, X, y, theta):
        '''
        Computes the objective function(cost function)
        Arguments:
          X is a n-by-d numpy matrix
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost  
              ** make certain you don't return a matrix with just one value! **
        '''
        n = len(y)
        predictions = self.predict(X, theta)
        cost = (1/2*n) * np.sum(np.square(predictions-y))
        return cost


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        # n = len(y)
        n,d = X.shape
        if self.theta is None:
            self.theta = np.matrix(np.zeros((d,1)))
        self.theta = self.gradientDescent(X,y,self.theta)

    def predict(self, X, theta):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        return np.dot(X,theta)
