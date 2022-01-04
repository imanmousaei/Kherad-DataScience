import numpy as np

from linreg import LinearRegression


if __name__ == "__main__":
    '''
        Main function to test multivariate linear regression
    '''
    
    # load the data
    filePath = 'data/multivariateData.dat'
    file = open(filePath,'r')
    allData = np.loadtxt(file, delimiter=',')

    X = np.matrix(allData[:,:-1])
    y = np.matrix((allData[:,-1])).T

    n,d = X.shape
    
    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    
    # Add a row of ones for the bias term
    X = np.c_[np.ones((n,1)), X]
    
    # initialize the model
    init_theta = np.matrix(np.random.randn((d+1))).T
    n_iter = 2000
    alpha = 0.01

    # Instantiate objects
    lr_model = LinearRegression(init_theta = init_theta, alpha = alpha, n_iter = n_iter)
    lr_model.fit(X,y)
    