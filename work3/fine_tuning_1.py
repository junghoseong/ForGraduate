import argparse
import contextlib
import os
import sys
import time
import pathlib
import pickle
from edgetpu.basic import basic_engine
from edgetpu.learn.backprop.softmax_regression import SoftmaxRegression
import numpy as np
from PIL import Image
@contextlib.contextmanager

def test_image(path):
  """Returns opened test image."""
  with open(path, 'rb') as f:
    with Image.open(f) as image:
      yield image

def save_label_map(label_map, out_path):
  """Saves label map to a file."""
  with open(out_path, 'w') as f:
    for key, val in label_map.items():
      f.write('%s %s\n' % (key, val))

def get_image_paths(data_dir):
  """Walks through data_dir and returns list of image paths and label map.
  Args:
    data_dir: string, path to data directory. It assumes data directory is
      organized as,
          - [CLASS_NAME_0]
            -- image_class_0_a.jpg
            -- image_class_0_b.jpg
            -- ...
          - [CLASS_NAME_1]
            -- image_class_1_a.jpg
            -- ...
  Returns:
    A tuple of (image_paths, labels, label_map)
    image_paths: list of string, represents image paths
    labels: list of int, represents labels
    label_map: a dictionary (int -> string), e.g., 0->class0, 1->class1, etc.
  """
  classes = None
  image_paths = []
  labels = []
  class_idx = 0
  for root, dirs, files in os.walk(data_dir):
    dirs.sort()
    if root == data_dir:
      # Each sub-directory in `data_dir`
      classes = dirs
    else:
      # Read each sub-directory
      assert classes[class_idx] in root
      print('Reading dir: %s, which has %d images' % (root, len(files)))
      for img_name in files:
        image_paths.append(os.path.join(root, img_name))
        labels.append(class_idx)
      class_idx += 1
  return image_paths, labels, dict(zip(range(class_idx), classes))

def shuffle_and_split(image_paths, labels, val_percent=0.0, test_percent=0.0):
  """Shuffles and splits data into train, validation, and test sets.
  Args:
    image_paths: list of string, of dim num_data
    labels: list of int of length num_data
    val_percent: validation data set percentage.
    test_percent: test data set percentage.
  Returns:
    Two dictionaries (train_and_val_dataset, test_dataset).
    train_and_val_dataset has the following fields.
      'data_train': data_train
      'labels_train': labels_train
      'data_val': data_val
      'labels_val': labels_val
    test_dataset has the following fields.
      'data_test': data_test
      'labels_test': labels_test
  """
  image_paths = np.array(image_paths)
  labels = np.array(labels)

  # Shuffle
  perm = np.random.permutation(image_paths.shape[0])
  image_paths = image_paths[perm]
  labels = labels[perm]

  # Split
  num_total = image_paths.shape[0]
  num_val = int(num_total * val_percent)
  num_test = int(num_total * test_percent)
  num_train = num_total - num_val - num_test

  train_and_val_dataset = {}
  train_and_val_dataset['data_train'] = image_paths[0:num_train]
  train_and_val_dataset['labels_train'] = labels[0:num_train]
  train_and_val_dataset['data_val'] = image_paths[num_train:num_train + num_val]
  train_and_val_dataset['labels_val'] = labels[num_train:num_train + num_val]
  
  test_dataset = {}
  test_dataset['data_test'] = image_paths[num_train + num_val:]
  test_dataset['labels_test'] = labels[num_train + num_val:]
  return train_and_val_dataset, test_dataset

def extract_embeddings(image_paths, engine):
  """Uses model to process images as embeddings.
  Reads image, resizes and feeds to model to get feature embeddings. Original
  image is discarded to keep maximum memory consumption low.
  Args:
    image_paths: ndarray, represents a list of image paths.
    engine: BasicEngine, wraps embedding extractor model.
  Returns:
    ndarray of length image_paths.shape[0] of embeddings.
  """
  _, input_height, input_width, _ = engine.get_input_tensor_shape()
  assert engine.get_num_of_output_tensors() == 1
  feature_dim = engine.get_output_tensor_size(0)
  embeddings = np.empty((len(image_paths), feature_dim), dtype=np.float32)
  for idx, path in enumerate(image_paths):
    with test_image(path) as img:
      img = img.resize((input_width, input_height), Image.NEAREST)
      _, embeddings[idx, :] = engine.run_inference(np.asarray(img).flatten())
      # _, embeddings[idx, :] = engine.RunInference(np.asarray(img).flatten())
  return embeddings

def part_A_run(model_A_path, data_dir, output_dir):
  """Output the intermediate embeddings, true labels, and label map given data and embedding extractor part A.
  Args:
    model_A_path: string, path to embedding extractor part A.
    data_dir: string, directory that contains data.
    output_dir: string, directory to save the intermediate embeddings, true labels, and label map.
  """
  t0 = time.perf_counter()

  # Preprocess training (and validation) data
  image_paths, labels, label_map = get_image_paths(data_dir)
  train_and_val_dataset, test_dataset = shuffle_and_split(image_paths, labels, val_percent=0.1, test_percent=0.1)
  # Initializes basic engine model here to avoid repeated initialization,
  # which is time consuming.
  engine = basic_engine.BasicEngine(model_A_path)
  print('Extract intermediate embeddings for data_train')
  train_and_val_dataset['data_train'] = extract_embeddings(train_and_val_dataset['data_train'], engine)
  print('Extract intermediate embeddings for data_val')
  train_and_val_dataset['data_val'] = extract_embeddings(train_and_val_dataset['data_val'], engine)
  print('Extract intermediate embeddings for data_test')
  test_dataset['data_test'] = extract_embeddings(test_dataset['data_test'], engine)
  t1 = time.perf_counter()
  print('Data preprocessing takes %.2f seconds' % (t1 - t0))

  with open(os.path.join(output_dir, 'train_and_val_dataset.pickle'), 'wb') as handle:
    pickle.dump(train_and_val_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open(os.path.join(output_dir, 'test_dataset.pickle'), 'wb') as handle:
    pickle.dump(test_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
  save_label_map(label_map, os.path.join(output_dir, 'label_map.txt'))
  t2 = time.perf_counter()
  print('Saving the intermediate embeddings, true labels, and label map takes %.2f seconds' % (t2 - t1))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--embedding_extractor_A_path', required=True, help='Path to embedding extractor part A tflite model which was compiled.')
  parser.add_argument('--dataset_dir', required=True, help='Directory to local dataset.')
  parser.add_argument('--output_dir', default='./output', help='Path to directory to save the intermediate embeddings, true labels, and label map.')
  
  args = parser.parse_args()
  if not os.path.exists(args.dataset_dir):
    sys.exit('%s does not exist!' % args.dataset_dir)
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  part_A_run(args.embedding_extractor_A_path, args.dataset_dir, args.output_dir)

if __name__ == '__main__':
  main()
