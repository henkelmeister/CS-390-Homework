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

def gradient_descent(X : np.ndarray, Y : np.ndarray, w0 : np.ndarray, alpha : float, num_iterations : int)->Tuple[List[float], np.ndarray]:
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
    w = np.copy(w0)
    losses = []
    # TODO Complete this function. Use previous homework if it was correct.
    return losses, w

def learning_curve(losses: list, names : list):
    """
    This function plots the learning curves for all gradient descent procedures in this homework.
    The plot is saved in the file learning_curve.png.

    Parameters
    ----------
    losses : list
        A list of arrays representing the losses for the gradient at each iteration for each run of gradient descent
    names : list
        A list of strings representing the names for each gradient descent method

    Returns
    -------
    nothing
    """
    for loss in losses:
        plt.plot(loss)
    plt.xscale("log")
    plt.ylim(0, 35000)
    plt.xlabel("Iterations")
    plt.ylabel("Squared Loss")
    plt.title("Gradient Descent")
    plt.legend(names)
    plt.savefig("learning_curve.png")
    plt.show()

def polynomial_basis(X: np.ndarray, order: int):
    """
    This function creates a new set of features comprised of polynomial features.

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array containing the existing features
    order : int
        A integer representing the order to use for the polynomial basis

    Returns
    -------
    X2 : np.ndarray
        A 2D numpy array containing the polynomial features
    """
    X2 = np.copy(X)  # TODO replace this line and complete the function
    return X2

def fourier_basis(X: np.ndarray, order : int):
    """
    This function creates a new set of features comprised of Fourier features.

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array containing the existing features
    order : int
        A integer representing the order to use for the Fourier basis

    Returns
    -------
    X2 : np.ndarray
        A 2D numpy array containing the Fourier features
    """
    X2 = np.copy(X)  # TODO replace the line and complete the function

    return X2

def add_constant(X: np.ndarray)->np.ndarray:
    """
    This function adds a column of ones to the matrix X. The column of ones will be the first column.
    The output of this function has m+1 columns if X has m columns.

    Parameters
    ----------
    X : np.ndarray
        2D numpy array containing data

    Returns
    -------
    X2 : np.ndarray
        A 2D numpy array with a column of ones in the first column.
    """
    X2 = np.copy(X)  # TODO replace this line to complete the fucntion
    m, n = np.shape(X)
    ones = np.ones((m , 1))
    X2 = np.append(ones,X2,axis=1)
    return X2

def normalize_gaussian(X: np.ndarray)->np.ndarray:
    """
    This function normalizes every column to be mean 0 with a standard deviation of 1

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array representing the data

    Returns
    -------
    X2 : np.ndarray
        A 2D numpy array with each column normalized
    """
    X2 = np.copy(X)  # TODO complete this function
    return X2

def normalize_01(X: np.ndarray, low: np.ndarray, high: np.ndarray)->np.ndarray:
    """
    This function normalizes every column to be in the range [0,1]

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array representing the data
    low : np.ndarray
        A 1D numpy array representing the minimum values for each column
    high : np.ndarray
        A 1D numpy array representing the maximum values for each column

    Returns
    -------
    X2 : np.ndarray
        A 2D numpy array with each column normalized
    """
    X2 = np.copy(X)  # TODO replace this line and complete the function
    return X2

def normalize_posneg(X: np.ndarray, low: np.ndarray, high: np.ndarray)->np.ndarray:
    """
    This function normalizes every column to be in the range [-1,1]

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array representing the data
    low : np.ndarray
        A 1D numpy array representing the minimum values for each column
    high : np.ndarray
        A 1D numpy array representing the maximum values for each column

    Returns
    -------
    X2 : np.ndarray
        A 2D numpy array with each column normalized
    """
    X2 = np.copy(X)  # TODO replace this line and complete the function
    return X2


def original_fit(X, Y, alpha, num_iterations=10000):
    X = add_constant(X)
    w = np.zeros(X.shape[1])  # initialize the weight vector
    losses, wfinal = gradient_descent(X, Y, w, alpha=alpha, num_iterations=num_iterations)
    return losses


def meanzero_fit(X, Y, alpha, num_iterations=10000):
    X = normalize_gaussian(X)
    X = add_constant(X)
    w = np.zeros(X.shape[1])  # initialize the weight vector
    losses, wfinal = gradient_descent(X, Y, w, alpha=alpha, num_iterations=num_iterations)
    return losses

def zeroone_fit(X, Y, low, high, alpha, num_iterations=10000):
    X = normalize_01(X, low, high)
    X = add_constant(X)
    w = np.zeros(X.shape[1])  # initialize the weight vector
    losses, wfinal = gradient_descent(X, Y, w, alpha=alpha, num_iterations=num_iterations)
    return losses

def posneg_fit(X, Y, low, high, alpha, num_iterations=10000):
    X = normalize_posneg(X, low, high)
    X = add_constant(X)
    w = np.zeros(X.shape[1])  # initialize the weight vector
    losses, wfinal = gradient_descent(X, Y, w, alpha=alpha, num_iterations=num_iterations)
    return losses

def fourier_fit(X: np.ndarray, Y: np.ndarray, order: int, low: np.ndarray, high: np.ndarray, alpha: float, num_iterations:int =10000):
    """
    This function performs a basis expansion using fourier features. Then
    performs gradient descent to find the minimum loss value. All loss values
    during gradient descent are returned. Feature normalization should be used
    before creating the fourier features and should be included in the final
    feature set. A bias feature should also be added to the expanded feature set.

    Parameters
    ----------
    X : np.ndarray
        A 2D array representing the original data
    Y : np.ndarray
        A 1D array representing the labels
    order : int
        The order for the fourier basis
    low : np.ndarray
        A 1D array containing the minimum feature values for each column
    high : np.ndarray
        A 1D array containing the maximum features values for each column
    alpha : float
        learning rate for gradient descent
    num_iterations : int
        number of iterations of gradient descent

    Returns
    -------
    losses : list
        A list of containing the loss values for gradient descent
    """
    losses = []  # TODO Complete this function
    return losses

def polynomial_fit(X: np.ndarray, Y: np.ndarray, order: int, low: np.ndarray, high: np.ndarray, alpha: float, num_iterations:int =10000):
    """
    This function performs a basis expansion using polynomial features. Then
    performs gradient descent to find the minimum loss value. All loss values
    during gradient descent are returned. Feature normalization should be used
    before creating the polynomial features. A bias feature should also be added.

    Parameters
    ----------
    X : np.ndarray
        A 2D array representing the original data
    Y : np.ndarray
        A 1D array representing the labels
    order : int
        The order for the polynomial basis
    low : np.ndarray
        A 1D array containing the minimum feature values for each column
    high : np.ndarray
        A 1D array containing the maximum features values for each column
    alpha : float
        learning rate for gradient descent
    num_iterations : int
        number of iterations of gradient descent

    Returns
    -------
    losses : list
        A list of containing the loss values for gradient descent
    """
    losses = []  # TODO complete this function
    return losses

def main():
    X, Y = load_data("hw4_data.csv")  # load the data set

    low = np.array([5.0, 0.0, 0.0, 30.0, -8.0])  # array containing the minimum values for each column in X
    high = np.array([10.0, 25.0, 0.5, 80.0, -2.5])  # array containing the maximum values for each column in X
    N = 10000  # N needs to equal 10,000 for your final plot. You can lower it to tune hyperparameters.


    losses0 = original_fit(X, Y, alpha=1e-7, num_iterations=N)  # performs gradient descent on the original data
    losses1 = meanzero_fit(X, Y, alpha=1e-7, num_iterations=N)  # performs gradient descent on the data with Gaussian normalization
    losses2 = zeroone_fit(X, Y, low, high, alpha=1e-7, num_iterations=N)  # performs gradient descent on the data with [0,1] normalization
    losses3 = posneg_fit(X, Y, low, high, alpha=1e-7, num_iterations=N)  # performs gradient descent on the data with [-1,1] normalization

    forder = 1  # order for the Fourier basis TODO tune this value
    porder = 2  # order for the Polynomial basis TODO tune this value
    losses4 = fourier_fit(X, Y, forder, low, high, alpha=1e-7, num_iterations=N)  # performs gradient descent on the data with fourier features added
    losses5 = polynomial_fit(X, Y, porder, low, high, alpha=1e-7, num_iterations=N)  # performs gradient descent on the data with polynomial features added


    all_losses = [losses0, losses1, losses2, losses3, losses4, losses5]
    names = ["Original", r"$\mathcal{N}(0,1)$", r"$[0,1]$", r"$[-1,1]$", "Fourier {0}".format(forder), "Poly {0}".format(porder)]
    print("final loss values")
    for name, losses in zip(names, all_losses):
        # print("{0:.<21}{1:>8.1f}".format(name, float(losses[-1])))

    learning_curve(all_losses, names)


if __name__ == "__main__":
    main()