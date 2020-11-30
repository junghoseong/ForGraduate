import os
import pickle

from edgetpu.learn.backprop import ops
from edgetpu.learn.utils import AppendFullyConnectedAndSoftmaxLayerToModel
import numpy as np

# Default names for weights and label map checkpoint.
_WEIGHTS_NAME = 'weights.pickle'
_LABEL_MAP_NAME = 'label_map.pickle'

class SoftmaxRegression(object):
  def __init__(self,
               feature_dim=None,
               num_classes=None,
               weight_scale=0.01,
               reg=0.0):
    """
    Args:
      feature_dim (int): The dimension of the input feature (length of the feature vector).
      num_classes (int): The number of output classes.
      weight_scale (float): A weight factor for computing new weights. The backpropagated
        weights are drawn from standard normal distribution, then multipled by this number to keep
        the scale small.
      reg (float): The regularization strength.
    """
    self.reg = reg
    self.feature_dim = feature_dim
    self.num_classes = num_classes
    self.label_map = None
    self.params = {}
    if feature_dim and num_classes:
      self.params['mat_w_fc1'] = weight_scale * np.random.randn(
          feature_dim, 32).astype(np.float32)
      self.params['vec_b_fc1'] = np.zeros((32,)).astype(np.float32)
      self.params['mat_w_fc2'] = weight_scale * np.random.randn(
          32, num_classes).astype(np.float32)
      self.params['vec_b_fc2'] = np.zeros((num_classes,)).astype(np.float32)

    # Needed to set proper quantization parameter for output tensor of FC layer.
    self.min_score = np.finfo(np.float32).max
    self.max_score = np.finfo(np.float32).min
    self.loss_history = []
    self.accuracy_history = []

  def _get_loss(self, mat_x, labels):
    """Calculates the loss of the current model for the given data, using a
    cross-entropy loss function.
    Args:
      mat_x (:obj:`numpy.ndarray`): The input data (image embeddings) to test, as a matrix of shape
        ``NxD``, where ``N`` is number of inputs to test and ``D`` is the dimension of the
        input feature (length of the feature vector).
      labels (:obj:`numpy.ndarray`): An array of the correct label indices that correspond to the
        test data passed in ``mat_x`` (class label index in one-hot vector). For example, if
        ``mat_x`` is just one image embedding, this array has one number for that image's correct
        label index.
    Returns:
      A 2-tuple with the cross-entropy loss (float) and gradients (a dictionary with ``'mat_w'``
      and ``'vec_b'``, for weight and bias, respectively).
    """
    mat_w_fc1 = self.params['mat_w_fc1']
    vec_b_fc1 = self.params['vec_b_fc1']
    mat_w_fc2 = self.params['mat_w_fc2']
    vec_b_fc2 = self.params['vec_b_fc2']

    scores_1, fc_cached_1 = ops.fc_forward(mat_x, mat_w_fc1, vec_b_fc1)
    scores_2, fc_cached_2 = ops.fc_forward(scores_1, mat_w_fc2, vec_b_fc2)
    # Record min, max value of scores.
    self.min_score = np.min([self.min_score, np.min(scores_2)])
    self.max_score = np.max([self.max_score, np.max(scores_2)])
    loss, dscores = ops.softmax_cross_entropy_loss(scores_2, labels)
    loss += 0.5 * self.reg * np.sum(mat_w_fc2 * mat_w_fc2)

    grads = {}
    dmat_x_fc2, grads['mat_w_fc2'], grads['vec_b_fc2'] = ops.fc_backward(dscores, fc_cached_2)
    dmat_x_fc1, grads['mat_w_fc1'], grads['vec_b_fc1'] = ops.fc_backward(dmat_x_fc2, fc_cached_1)
    #grads['mat_w'] += self.reg * mat_w

    return loss, grads

  def run_inference(self, mat_x):
    """Runs an inference using the current weights.
    Args:
      mat_x (:obj:`numpy.ndarray`): The input data (image embeddings) to infer, as a matrix of shape
        ``NxD``, where ``N`` is number of inputs to infer and ``D`` is the dimension of the
        input feature (length of the feature vector). (This can be one or more image embeddings.)
    Returns:
      The inferred label index (or an array of indices if multiple embeddings given).
    """
    mat_w_fc1 = self.params['mat_w_fc1']
    vec_b_fc1 = self.params['vec_b_fc1']
    mat_w_fc2 = self.params['mat_w_fc2']
    vec_b_fc2 = self.params['vec_b_fc2']
    
    scores, _ = ops.fc_forward(mat_x, mat_w_fc1, vec_b_fc1)
    scores, _ = ops.fc_forward(scores, mat_w_fc2, vec_b_fc2)
    
    if len(scores.shape) == 1:
      return np.argmax(scores)
    else:
      return np.argmax(scores, axis=1)

  def save_as_tflite_model(self, in_model_path, out_model_path):
    """Appends learned weights to your TensorFlow Lite model and saves it as a copy.
    Beware that learned weights and biases are quantized from float32 to uint8.
    Args:
      in_model_path (str): Path to the embedding extractor model (``.tflite`` file).
      out_model_path (str): Path where you'd like to save the new model with learned weights
        and a softmax layer appended (``.tflite`` file).
    """
    # Note: this function assumes flattened weights, whose dimension is
    # num_classes x feature_dim. That's why the transpose is needed.
    print(self.loss_history)
    print(self.accuracy_history)
    AppendFullyConnectedAndSoftmaxLayerToModel(
        in_model_path, out_model_path,
        self.params['mat_w_fc1'].transpose().flatten(),
        self.params['vec_b_fc1'].flatten(), 
        self.params['mat_w_fc2'].transpose().flatten(),
        self.params['vec_b_fc2'].flatten(), float(self.min_score),
        float(self.max_score))

  def get_accuracy(self, mat_x, labels):
    """Calculates the model's accuracy (percentage correct) when performing inferences on the
    given data and labels.
    Args:
      mat_x (:obj:`numpy.ndarray`): The input data (image embeddings) to test, as a matrix of shape
        ``NxD``, where ``N`` is number of inputs to test and ``D`` is the dimension of the
        input feature (length of the feature vector).
      labels (:obj:`numpy.ndarray`): An array of the correct label indices that correspond to the
        test data passed in ``mat_x`` (class label index in one-hot vector).
    Returns:
      The accuracy (the percent correct) as a float.
    """
    return np.mean(self.run_inference(mat_x) == labels)

  def train_with_sgd(self,
                     data,
                     num_iter,
                     learning_rate,
                     batch_size=100,
                     print_every=100):
    """Trains your model using stochastic gradient descent (SGD).
    The training data must be structured in a dictionary as specified in the ``data`` argument
    below. Notably, the training/validation images must be passed as image embeddings, not as the
    original image input. That is, run the images through your embedding extractor
    (the backbone of your graph) and use the resulting image embeddings here.
    Args:
      data (dict): A dictionary that maps ``'data_train'`` to an array of training image embeddings,
        ``'labels_train'`` to an array of training labels, ``'data_val'`` to an array of validation
        image embeddings, and ``'labels_val'`` to an array of validation labels.
      num_iter (int): The number of iterations to train.
      learning_rate (float): The learning rate (step size) to use in training.
      batch_size (int): The number of training examples to use in each iteration.
      print_every (int): The number of iterations for which to print the loss, and
        training/validation accuracy. For example, ``20`` prints the stats for every 20 iterations.
        ``0`` disables printing.
    """
    data_train = data['data_train']
    labels_train = data['labels_train']
    data_val = data['data_val']
    labels_val = data['labels_val']
    mat_w = self.params['mat_w_fc2']
    vec_b = self.params['vec_b_fc2']
    num_train = data_train.shape[0]

    for i in range(num_iter):
      batch_mask = np.random.choice(num_train, batch_size)
      data_batch = data_train[batch_mask]
      labels_batch = labels_train[batch_mask]
      loss, grads = self._get_loss(data_batch, labels_batch)
      # Simple SGD update rule.
      mat_w -= learning_rate * grads['mat_w_fc2']
      vec_b -= learning_rate * grads['vec_b_fc2']

      if (print_every > 0) and (i % print_every == 0):
        print('Loss %.2f, train acc %.2f%%, val acc %.2f%%' %
              (loss, 100 * self.get_accuracy(data_train, labels_train),
               100 * self.get_accuracy(data_val, labels_val)))
        self.loss_history.append(loss)
        self.accuracy_history.append(self.get_accuracy(data_val, labels_val))

  def _set_label_map(self, label_map):
    """Attaches label_map with the model."""
    self.label_map = label_map

  def _get_label_map(self):
    """Gets label_map with the model."""
    return self.label_map

  def _save_ckpt(self, ckpt_dir):
    """Saves checkpoint."""
    if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)
    # Save weights
    weights_path = os.path.join(ckpt_dir, _WEIGHTS_NAME)
    with open(weights_path, 'wb') as fp:
      pickle.dump(self.params, fp)
    # Save label map
    if self.label_map:
      label_map_path = os.path.join(ckpt_dir, _LABEL_MAP_NAME)
      with open(label_map_path, 'wb') as fp:
        pickle.dump(self.label_map, fp)

  def _load_ckpt(self, ckpt_dir):
    """Loads weights and label_map from file."""
    weights_path = os.path.join(ckpt_dir, _WEIGHTS_NAME)
    with open(weights_path, 'rb') as fp:
      self.params = pickle.load(fp)
    self.feature_dim = self.params['mat_w_fc1'].shape[0]
    self.num_classes = self.params['mat_w_fc2'].shape[1]

    label_map_path = os.path.join(ckpt_dir, _LABEL_MAP_NAME)
    with open(label_map_path, 'rb') as fp:
      self.label_map = pickle.load(fp)