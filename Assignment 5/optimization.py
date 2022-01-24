import numpy as np


class Regressor:

    def __init__(self) -> None: 
        self.X, self.y = self.generate_dataset(n_samples=200, n_features=1)
        n, d = self.X.shape
        self.w = np.zeros((d, 1))

    def generate_dataset(self, n_samples, n_features):
        """
        Generates a regression dataset
        Returns:
            X: a numpy.ndarray of shape (100, 2) containing the dataset
            y: a numpy.ndarray of shape (100, 1) containing the labels
        """
        from sklearn.datasets import make_regression
        
        np.random.seed(42)
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=30)
        y = y.reshape(n_samples, 1)
        return X, y


    def linear_regression(self):
        """
        Performs linear regression on a dataset
        Returns:
            y: a numpy.ndarray of shape (n, 1) containing the predictions
        """
        y = np.dot(self.X, self.w)
        return y


    def predict(self, X):
        """
        Predicts the labels for a given dataset
        X: a numpy.ndarray of shape (n, d) containing the dataset
        Returns:
            y: a numpy.ndarray of shape (n,) containing the predictions
        """
        y = np.dot(X, self.w).reshape(X.shape[0])
        return y


    def compute_loss(self):
        """
        Computes the MSE loss of a prediction
        Returns:
            loss: the loss of the prediction
        """
        predictions = self.linear_regression()
        loss = np.mean((predictions - self.y)**2)
        return loss


    def compute_gradient(self):
        """
        Computes the gradient of the MSE loss
        Returns:
            grad: the gradient of the loss with respect to w
        """
        predictions = self.linear_regression()
        dif = (predictions - self.y)
        grad = 2 * np.dot(self.X.T, dif)
        return grad


    def fit(self, optimizer='gd', n_iters=1000, render_animation=False):
        """
        Trains the model
        optimizer: the optimization algorithm to use
        X: a numpy.ndarray of shape (n, d) containing the dataset
        y: a numpy.ndarray of shape (n, 1) containing the labels
        n_iters: the number of iterations to train for
        """        

        figs = []

        for i in range(1, n_iters+1):

            if optimizer == 'gd':
                # TODO: implement gradient descent
                pass
            elif optimizer == "sgd":
                # TODO: implement stochastic gradient descent
                pass
            elif optimizer == "sgdMomentum":
                # TODO: Implement the SGD with momentum
                pass
            elif optimizer == "adagrad":
                # TODO: Implement Adagrad
                pass
            elif optimizer == "rmsprop":
                # TODO: implement RMSprop
                pass
            elif optimizer == "adam":
                # TODO: implement Adam optimizer
                pass

            # TODO: implement the stop criterion
            
            if i % 10 == 0:
                print("Iteration: ", i)
                print("Loss: ", self.compute_loss())
            
            if render_animation:
                import matplotlib.pyplot as plt
                from moviepy.video.io.bindings import mplfig_to_npimage

                fig = plt.figure()
                plt.scatter(self.X, self.y, color='red')
                plt.plot(self.X, self.predict(self.X), color='blue')
                plt.xlim(self.X.min(), self.X.max())
                plt.ylim(self.y.min(), self.y.max())
                plt.title(f'Optimizer:{optimizer}\nIteration: {i}')
                plt.close()
                figs.append(mplfig_to_npimage(fig))
            
        
        if render_animation and len(figs) > 0:
            from moviepy.editor import ImageSequenceClip
            clip = ImageSequenceClip(figs, fps=5)
            clip.write_gif(f'{optimizer}_animation.gif', fps=5)


    def gradient_descent(self, alpha):
        """
        Performs gradient descent to optimize the weights
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        w = None
        # TODO: implement gradient descent

        return w


    def sgd_optimizer(self, alpha):
        """
        Performs stochastic gradient descent to optimize the weights
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        w = None
        # TODO: implement stochastic gradient descent

        return w


    def sgd_momentum(self, alpha=0.01, momentum=0.9):
        """
        Performs SGD with momentum to optimize the weights
        alpha: the learning rate
        momentum: the momentum
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        w = None
        # TODO: implement stochastic gradient descent

        return w

    
    def adagrad_optimizer(self, g, alpha, epsilon):
        """
        Performs Adagrad optimization to optimize the weights
        alpha: the learning rate
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        w = None
        # TODO: implement stochastic gradient descent
        
        return w

    
    def rmsprop_optimizer(self, g, alpha, beta, epsilon):
        """
        Performs RMSProp optimization to optimize the weights
        g: sum of squared gradients
        alpha: the learning rate
        beta: the momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        w = None
        # TODO: implement stochastic gradient descent
        
        return w


    def adam_optimizer(self, m, v, alpha, beta1, beta2, epsilon):
        """
        Performs Adam optimization to optimize the weights
        m: the first moment vector
        v: the second moment vector
        alpha: the learning rate
        beta1: the first momentum
        beta2: the second momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        w = None
        # TODO: implement stochastic gradient descent
        
        return w


    def plot_gradient():
        """
        Plots the gradient descent path for the loss function
        Useful links: 
        -   http://www.adeveloperdiary.com/data-science/how-to-visualize-gradient-descent-using-contour-plot-in-python/
        -   https://www.youtube.com/watch?v=zvp8K4iX2Cs&list=LL&index=2
        """
        # TODO: Bonus!
        pass