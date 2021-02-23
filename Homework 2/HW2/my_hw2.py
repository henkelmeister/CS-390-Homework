import numpy as np
from typing import Tuple


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


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    '''
    This function makes a prediction for each set of features in X using
    weights w.

    Parameters
    ----------
    X : np.ndarray
        A 2D array of shape (n,m) where each row is a different set of features
    w : np.ndarray
        A 1D array of shape (m,) containing the weights for the linear model

    Returns
    -------
    predictions : np.ndarray
        A 1D array of shape (n,) containing the predictions for each set of
        features in X.
    '''
    n, m = X.shape  # get number of feature vectors and length of the feature vectors
    #predictions = np.zeros(n)  # initialize the list of predictions

    # TODO make a prediction for each feature vector X[i] and store it in predictions[i]

    """
    for i in range(0, n):
        sum = 0
        for j in range(0, m):
            sum += X[i][j] * w[j]
        predictions[i] = sum
    """
    """
    n, m = X.shape  # get number of feature vectors and length of the feature vectors
    predictions = np.zeros(n)  # initialize the list of predictions
    for i in range(0, n):
        sum = 0
        for j in range(0, m):
            sum += X[i][j] * w[j]
        predictions[i] = sum
    """
    predictions = X.dot(w)
    print("damn it")
    return predictions


def main():
    X, Y = load_data("hw2_data.csv")  # load the data set
    arr = []
    n, m = X.shape
    for x in range(0, m):
        arr.append(x)
    w = np.array(arr)  # TODO replace None with your initialization of w
    yhat = predict(X, w)  # predict the labels for X using w

    print("Predictions")
    print(yhat)
    print("Labels")
    print(Y)

    return w


if __name__ == "__main__":
    main()

