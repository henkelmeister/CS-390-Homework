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

def predict(X: np.ndarray, w: np.ndarray)->np.ndarray:
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
    n,m = X.shape  # get number of feature vectors and length of the feature vectors
    predictions = np.zeros(n)  # initialize the list of predictions

    for i in range(n):
        x = X[i]  # get the ith row
        yhat = 0  # initialize the prediction
        for j in range(m):  # make the prediction for the ith feature vector
            yhat +=  w[j] * x[j]  # multiply each element of the feature vector with the weight vector
        predictions[i] = yhat  # update the list of predictions

    return predictions

def predict_fast(X: np.ndarray, w: np.ndarray)->np.ndarray:
    '''
    This function makes a prediction for each set of features in X using
    weights w. This computation is faster the the predict function by using
    numpy dot product function.

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
    predictions = np.zeros(n)  # initialize the list of predictions

    for i in range(n):
        x = X[i]  # get the ith row
        predictions[i] = np.multiply(x, w)

    return predictions

def predict_faster(X: np.ndarray, w: np.ndarray)->np.ndarray:
    '''
    This function makes a prediction for each set of features in X using
    weights w. This computation is faster the the predict function by using
    matrix-vector multiplication. This is faster because for loops are slow
    in python.

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
    predictions = np.dot(X, w)
    print(predictions)

    return predictions


def main():
    X, Y = load_data("hw2_data.csv")  # load the data set
    w = np.arange(X.shape[1]) # replace None with your initialization of w
    yhat = predict(X, w) # predict the labels for X using w

    print("Predictions")
    print(yhat)
    print("Labels")
    print(Y)

    return w



if __name__ == "__main__":
    main()