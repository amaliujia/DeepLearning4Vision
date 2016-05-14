import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,y[i]] -= X[i, :].T;
        dW[:, j] += X[i, :].T;

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_train = X.shape[0]
  scores = X.dot(W)

  correct_scores = np.zeros(scores.shape)
  for i in xrange(num_train):
    correct_scores[i, :] = scores[i, y[i]]

  margins_mat = scores - correct_scores + 1
  margins_mat[np.arange(num_train), y] = 0
  valid = np.maximum(np.zeros(margins_mat.shape), margins_mat)
  
  loss = np.sum(valid)
  loss /= num_train 
  loss += 0.5 * reg * np.sum(W * W)

  binary_indicator = valid
  binary_indicator[valid > 0] = 1

  row_sum = np.sum(binary_indicator, axis=1)
  binary_indicator[range(num_train), y] -= row_sum[range(num_train)]
  #binary_indicator[range(num_train), y] += row_sum[range(num_train)]
  dW = np.dot(X.T, binary_indicator)

  # Where is the regularization on weights
  dW /= num_train
  dW += reg * W
  return loss, dW
