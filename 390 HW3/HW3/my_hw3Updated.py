import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

def load_data(file_path: str)->Tuple[np.ndarray, np.ndarray]:
    '''
    This function loads and parses text file separated by a ',' character and
    returns a data set as two arrays, an array of features, and an array of labels.

    Parameters
    ----------
    file_path : str
                path to the file containing the data set

    Returns
    -------
    features : ndarray
                2D array of shape (n,m) containing features for the data set
    labels : ndarray
                1D array of shape (n,) containing labels for the data set
    '''
    D = np.genfromtxt(file_path, delimiter=",")
    features = D[:, :-1]  # all columns but the last one
    labels = D[:, -1]  # the last column
    return features, labels

def loss_function(X : np.ndarray, Y : np.ndarray, w : np.ndarray)->float:
    """
    This function computes the squared loss (scaled by 0.5) for predictions of
    a linear model using weights w on a data set with feature vectors X and
    labels Y.

    Parameters
    ----------
    X : np.ndarray
        2D numpy array where features vectors are on each row
    Y : np.ndarray
        1D numpy array containing the labels for each feature vector
    w : np.ndarray
        1D nuumpy array representing the weight vector

    Returns
    -------
    loss : float
        value representing the loss
    """
    """
    loss = 0.0
    n, m = np.shape(X)
    predictions = np.zeros(X.shape[0])

    for i in range(0, n):
        for j in range(0, m):
            predictions[i] = (X[i][j] * w[j])
        loss += .5*((predictions[i] - Y[i])**2)
    print("Loss is {}".format(loss))
    """
    # TODO [optional] complete this function

    y_estimated = X.dot(w)
    error = y_estimated - Y
    loss = .5*np.sum(error**2)


    return loss

def loss_gradient(X : np.ndarray, Y : np.ndarray, w : np.ndarray)->Tuple[float, np.ndarray]:
    """
    This function computes the squared loss (scaled by 0.5) for predictions of
    a linear model using weights w on a data set with feature vectors X and
    labels Y. It additionally computes and returns the gradient of the loss
    function with respect to the weights w.

    Parameters
    ----------
    X : np.ndarray
        2D numpy array where features vectors are on each row
    Y : np.ndarray
        1D numpy array containing the labels for each feature vector
    w : np.ndarray
        1D nuumpy array representing the weight vector

    Returns
    -------
    loss : float
        value representing the loss
    g : np.ndarray
        1D numpy array representing the gradient
    """
    """
    m, n = X.shape
    loss = loss_function(X, Y, w)
    #g = np.zeros_like(w)  # makes an array of zeros in the same shape as w
    #g = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    for i in range(0, n):
        for j in range(0, m):
            g[i] += (X[i][j]*w[i] - Y[i])*X[i][j]
    
    """
    #  print("Gradient is {}".format(g))
    # TODO [optional] complete this function
    loss = loss_function(X, Y, w)
    y_estimated = X.dot(w)
    error = (y_estimated - Y)
    g = X.T.dot(error)

    return loss, g


def gradient_descent(X: np.ndarray, Y: np.ndarray, w0: np.ndarray, alpha: float, num_iterations: int) -> Tuple[List[float], np.ndarray]:
    """
    This function runs gradient descent for a fixed number of iterations on the
    mean squared loss for a linear model parameterized by the weight vector w.
    This function returns a list of the losses for each iteration of gradient
    descent along with the final weight vector.

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array representing where each row represents a feature vector
    Y : np.ndarray
        A 1D numpy array where each element represents a label for MSE
    w0 : np.ndarray
        A 1D numpy array representing the initial weight vector
    alpha : float
        The step-size parameter to use with gradient descent.
    num_iterations : int
        The number of iterations of gradient descent to run.

    Returns
    -------
    losses : list
        A list of floats representing the loss from each iteration and the
        loss of the final weight vector
    w : np.ndarray
        The final weight vector produced by gradient descent.

    """
    threshold = .05
    prevLoss = 1
    currLoss = 0
    w = np.copy(w0)
    losses = []
    n, m = X.shape
    # TODO complete this function
    testNum = 1
    i = 0
    while (i < num_iterations) and (abs(prevLoss - currLoss) > threshold):
        loss, g = loss_gradient(X, Y, w)
        prevLoss = currLoss
        currLoss = loss
        for j in range(0, m):
            w[j] = w[j] - (alpha*g[j])
        print(loss)
        # w = w + (alpha * g) * X
        losses.append(loss)
        i += 1
    print("The last iteration was at {}".format(i))

    return losses, w


def learning_curve(losses: List[float], kloss : float):
    """
    This function plots the learning curve for gradient descent and a line
    representing the loss for the k-NN model. The plot is saved in the file
    learning_curve.png.

    Parameters
    ----------
    losses : list
        A list of floats representing the losses for the gradient at each iteration
    kloss : float
        The loss for the k-NN model

    Returns
    -------
    None
    """

    plt.plot(losses, color="dodgerblue")
    plt.hlines(kloss, 0, len(losses), color="crimson", linestyle="dashed")
    plt.xlabel("Iterations")
    plt.ylabel("Squared Loss")
    plt.title("Gradient Descent")
    plt.legend(["Linear Model", "k-NN"])
    plt.ylim(0, 35*1052)
    plt.savefig("learning_curve.png")
    plt.show()


def knn_mse(X: np.ndarray, Y: np.ndarray, k: int) -> float:
    """
    This function computes the mean squared loss on a data set for a k-NN
    model.

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array where each row contains the feature vectors
    Y : np.ndarray
        A 1D nunmpy array where each element is the label
    k : int
        number of nearest neighbors. Must be >= 1

    Returns
    -------
    loss : float
        the mean squared loss for the k-NN model
    """
    assert k >= 1
    A = X.reshape(X.shape[0], 1, X.shape[1])
    D = np.sqrt(np.square(X - A).sum(axis=-1))
    idxs = np.argsort(D, axis=-1)
    Yhat = Y[idxs[:, 1:k+1]].mean(axis=-1)
    loss = np.square(Yhat - Y).sum() * 0.5
    return loss

def main():
    """
    This function runs gradient descent on the data set contained in hw3_data.csv
    and it computes the loss for a k-NN model on the same data set. It then plots
    the learning curve and returns the final loss for each model.

    Returns
    -------
    linear_regression_loss : float
        The final loss value for the weights returned by gradient descent
    kloss : float
        The loss value for the k-NN model
    """
    X, Y = load_data("hw3_data.csv")  # load the data set
    alpha = 0.000847  # TODO find an optimal value of this value
    #alpha = 0.000747
    num_iterations = 1000  # TODO find an optimal value of this value
    w = np.zeros(X.shape[1])  # initialize the weight vector
    losses, wfinal = gradient_descent(X, Y, np.copy(w), alpha=alpha, num_iterations=num_iterations)

    k = 55  # number of nearest neighbors TODO find the optimal value for this value
    kloss = knn_mse(X, Y, k)

    learning_curve(losses, kloss)  # plot the learning curve for gradient descent

    linear_regression_loss = losses[-1]
    print("final gradient descent loss: {0:.3f}, k-NN loss {1:.3f}".format(linear_regression_loss, kloss))

    return linear_regression_loss, kloss


if __name__ == "__main__":
    main()