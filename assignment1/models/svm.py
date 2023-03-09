"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
  def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
    """Initialize a new classifier.

    Parameters:
        n_class: the number of classes
        lr: the learning rate
        epochs: the number of epochs to train for
        reg_const: the regularization constant
    """
    self.w = None  # TODO: change this
    self.lr = lr
    self.epochs = epochs
    self.reg_const = reg_const
    self.n_class = n_class
    self.batch_size = 256

  def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Calculate gradient of the svm hinge loss.

    Inputs have dimension D, there are C classes, and we operate on
    mini-batches of N examples.

    Parameters:
        X_train: a numpy array of shape (N, D) containing a mini-batch
            of data
        y_train: a numpy array of shape (N,) containing training labels;
            y[i] = c means that X[i] has label c, where 0 <= c < C

    Returns:
        the gradient with respect to weights w; an array of the same shape
            as w
    """
    # TODO: implement me
    N, D = X_train.shape
    C = self.n_class
    grad = np.zeros((C, D))

    for i in range(N):
      scores = self.w @ X_train[i]
      correct_class_score = scores[y_train[i]]
      for j in range(C):
        if j == y_train[i]:
          continue
        margin = scores[j] - correct_class_score + 1
        if margin > 0:
          grad[j, :] += X_train[i, :]
          grad[y_train[i], :] -= X_train[i, :]

    grad /= N
    grad += 2 * self.reg_const * self.w

    return grad

  def train(self, X_train: np.ndarray, y_train: np.ndarray):
    """Train the classifier.

    Hint: operate on mini-batches of data for SGD.

    Parameters:
        X_train: a numpy array of shape (N, D) containing training data;
            N examples with D dimensions
        y_train: a numpy array of shape (N,) containing training labels
    """
    # TODO: implement me
    N, D = X_train.shape
    C = self.n_class
    self.w = np.random.randn(C, D)
    num_batch = N // self.batch_size

    for epoch in range(self.epochs):
      for i in range(num_batch):
        idx = np.random.choice(N, self.batch_size, replace=False)
        x_i = X_train[idx]
        y_i = y_train[idx]
        grad = self.calc_gradient(x_i, y_i)
        self.w -= self.lr * grad
      self.lr *= np.exp(-1 * (epoch))

  def predict(self, X_test: np.ndarray) -> np.ndarray:
    """Use the trained weights to predict labels for test data points.

    Parameters:
        X_test: a numpy array of shape (N, D) containing testing data;
            N examples with D dimensions

    Returns:
        predicted labels for the data in X_test; a 1-dimensional array of
            length N, where each element is an integer giving the predicted
            class.
    """
    # TODO: implement me
  
    pred = np.dot(X_test, self.w.T)
    return np.argmax(pred, axis=1) # min
