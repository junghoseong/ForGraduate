import numpy as np


def fc_forward(mat_x, mat_w, vec_b):
  """Forward pass of Fully-Connected layer.
  A good reference for this is: http://cs231n.github.io/linear-classify/#score
  Args:
    mat_x: NxD ndarray, N is number of features, D is length of feature vector
    mat_w: DxC ndarray, C is number of classes.
    vec_b: length C ndarray.
  Returns:
    a tuple of (mat_out, cached)
    mat_out: NxC ndarray, as defined by Y=X*W+b.
    cached: value stored to help calculating gradient in backward pass.
  """
  mat_out = mat_x.dot(mat_w) + vec_b
  cached = (mat_x, mat_w)
  return mat_out, cached


def fc_backward(dout, cached):
  """Backward pass of Fully-Connected layer.
  FC layer is defined by: Y=X*W+b
  In general, the gradient of a function, which has tensor input and output, is
  a high dimensional tensor. But for linear relation as Y=X*W+b, this high
  dimensional tensor has a lot of zeros and can be simplified.
  Args:
    dout: NxC ndarray, gradient with respect to Y
    cached: cached value from fc_forward
  Returns:
    a tuple of gradients with respect to X, W, b
  """
  mat_x, mat_w = cached
  dmat_x = dout.dot(mat_w.T)
  dmat_w = mat_x.T.dot(dout)
  dvec_b = dout.T.dot(np.ones([mat_x.shape[0]]))
  return dmat_x, dmat_w, dvec_b


def softmax_cross_entropy_loss(logits, labels):
  """
  Args:
    logits: NxC ndarray, unnormalized logits
    labels: length N ndarray, index of class label in one hot vector.
  Returns:
    A tuple of Cross-entropy loss and gradient with respect to logits
  """
  # Use softmax(x) = softmax(x-C) to avoid exp() overflow.
  logits -= np.max(logits, axis=1, keepdims=True)
  probs = np.exp(logits)
  probs /= np.sum(probs, axis=1, keepdims=True)
  num_input = logits.shape[0]
  loss = -np.sum(np.log(probs[range(num_input), labels])) / num_input

  dlogits = probs.copy()
  dlogits[range(num_input), labels] -= 1
  dlogits /= num_input
  return loss, dlogits