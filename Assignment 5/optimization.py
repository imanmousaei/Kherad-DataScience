import numpy as np


class Regressor:

    def __init__(self) -> None:
        self.X, self.y = self.generate_dataset(n_samples=200, n_features=1)
        n, d = self.X.shape
        self.w = np.zeros((d, 1))
        self.n = n
        self.d = d


    def generate_dataset(self, n_samples, n_features):
        """
        Generates a regression dataset
        Returns:
            X: a numpy.ndarray of shape (100, 2) containing the dataset
            y: a numpy.ndarray of shape (100, 1) containing the labels
        """
        from sklearn.datasets import make_regression

        np.random.seed(42)
        X, y = make_regression(n_samples=n_samples,
                               n_features=n_features, noise=30)
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


    def fit(self, optimizer='sgd', n_iters=1000, render_animation=False):
        """
        Trains the model
        optimizer: the optimization algorithm to use; enum:[sgd, sgdMomentum, adagrad, rmsprop, adam]
        X: a numpy.ndarray of shape (n, d) containing the dataset
        y: a numpy.ndarray of shape (n, 1) containing the labels
        n_iters: the number of iterations to train for
        """

        optimizer = optimizer.lower()
        figs = []
        grads = self.compute_gradient()

        for i in range(1, n_iters+1):
            w = 0

            if optimizer == "sgd":
                w = self.sgd_optimizer(grads)
            elif optimizer == "sgdmomentum":
                w = self.sgd_momentum(grads)
            elif optimizer == "adagrad":
                w = self.adagrad_optimizer(grads)
            elif optimizer == "rmsprop":
                w = self.rmsprop_optimizer(grads)
            elif optimizer == "adam":
                w = self.adam_optimizer(grads)

            self.w = w

            # the stop criterion
            if self.compute_loss() < 0.1:
                break

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


    def sgd_optimizer(self, grads, alpha=0.1):
        """
        Performs stochastic gradient descent to optimize the weights
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """

        updated_w = []

        for param, grad in zip(self.w, grads):
            delta = alpha * grad
            param -= delta

            updated_w.append(param)

        return updated_w


    def sgd_momentum(self, grads, alpha=0.01, momentum=0.9):
        """
        Performs SGD with momentum to optimize the weights
        momentum: the momentum
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        updated_w,  prevs = [], []

        for param, grad, prev_update in zip(self.w, grads, self.previous_updates):
            delta = alpha * grad - momentum * prev_update
            param -= delta

            prevs.append(delta)
            updated_w.apppend(param)

        self.previous_updates = prevs

        return updated_w


    def adagrad_optimizer(self, grads, alpha=0.01, epsilon=1e-6):
        """
        Performs Adagrad optimization to optimize the weights
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        updated_w = []

        for i, (param, grad) in enumerate(zip(self.w, grads)):
            self.cache[i] += grad ** 2
            param += -alpha * grad / \
                (np.sqrt(self.cache[i]) + epsilon)

            updated_w.apppend(param)

        return updated_w

    def rmsprop_optimizer(self, grads, beta=0.9, alpha=0.001, epsilon=1e-6):
        """
        Performs RMSProp optimization to optimize the weights
        g: sum of squared gradients
        beta: the momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        updated_w = []
        history = [0] * self.d

        for i, (param, grad) in enumerate(zip(self.w, grads)):
            history[i] = beta * history[i] + (1-beta) * (grad ** 2)
            param += - alpha * grad / (np.sqrt(history[i]) + epsilon)

            updated_w.append(param)

        return updated_w


    def adam_optimizer(self, grads, beta1=0.9, beta2=0.999, alpha=0.001, epsilon=1e-8):
        """
        Performs Adam optimization to optimize the weights
        m: the first moment vector
        v: the second moment vector
        beta1: the first momentum
        beta2: the second momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        updated_w = []
        m = v = [0] * self.d
        t = 1

        for i, (param, grad) in enumerate(zip(self.w, grads)):
            m[i] = beta1 * m[i] + (1-beta1) * grad
            v[i] = beta2 * v[i] + (1-beta2) * (grad ** 2)
            m_corrected = m[i] / (1-beta1**t)
            v_corrected = v[i] / (1-beta2**t)
            param += -alpha * m_corrected / (np.sqrt(v_corrected) + epsilon)
            updated_w.append(param)

        t += 1

        return updated_w


    def plot_gradient():
        """
        Plots the gradient descent path for the loss function
        Useful links: 
        -   http://www.adeveloperdiary.com/data-science/how-to-visualize-gradient-descent-using-contour-plot-in-python/
        -   https://www.youtube.com/watch?v=zvp8K4iX2Cs&list=LL&index=2
        """
        # TODO: Bonus!
        pass
