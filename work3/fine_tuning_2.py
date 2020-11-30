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

def extract_final_embeddings(im_embeddings, engine, scaling_factor, zero_point):
  # _, input_height, input_width, _ = engine.get_input_tensor_shape()
  assert engine.get_num_of_output_tensors() == 1
  feature_dim = engine.get_output_tensor_size(0)
  embeddings = np.empty((len(im_embeddings), feature_dim), dtype=np.float32)
  for idx, im_embedding in enumerate(im_embeddings):
    aa = np.asarray(im_embedding) / scaling_factor
    aa = aa + zero_point
    aa = np.asarray(aa, dtype=np.uint8).flatten()
    if idx == 2:
        np.savetxt('./tempp.txt', aa)
    _, embeddings[idx, :] = engine.run_inference(aa)

  return embeddings

def part_B_run_and_train(model_B_path, data_dir, output_dir, scaling_factor, zero_point):
  """Trains a softmax regression model given the intermediate embeddings, true labels, label map and embedding extractor part B.
  Args:
    model_B_path: string, path to embedding extractor part B.
    data_dir: string, directory that contains the intermediate embeddings, true labels, label map.
    output_dir: string, directory to save retrained tflite model and label map.
  """
  t0 = time.perf_counter()

  # Load the the intermediate embeddings and true labels
  with open(os.path.join(data_dir, 'train_and_val_dataset.pickle'), 'rb') as handle:
    train_and_val_dataset = pickle.load(handle)
  with open(os.path.join(data_dir, 'test_dataset.pickle'), 'rb') as handle:
    test_dataset = pickle.load(handle)

  # Initializes basic engine model here to avoid repeated initialization,
  # which is time consuming.
  engine = basic_engine.BasicEngine(model_B_path)
  print('Extract embeddings for data_train')
  train_and_val_dataset['data_train'] = extract_final_embeddings(train_and_val_dataset['data_train'], engine, scaling_factor, zero_point)
  print('Extract embeddings for data_val')
  train_and_val_dataset['data_val'] = extract_final_embeddings(train_and_val_dataset['data_val'], engine, scaling_factor, zero_point)
  t1 = time.perf_counter()
  print('Data preprocessing takes %.2f seconds' % (t1 - t0))

  # Construct FC + softmax and start training
  weight_scale = 5e-2
  reg = 0.0
  feature_dim = train_and_val_dataset['data_train'].shape[1]
  print('feature_dim: %d'%feature_dim)
  num_classes = np.max(train_and_val_dataset['labels_train']) + 1
  print('num_classes: %d'%num_classes)
  model = SoftmaxRegression(feature_dim, num_classes, weight_scale=weight_scale, reg=reg)
  learning_rate = 1e-2
  batch_size = 100
  num_iter = 500
  label_map = " "
  with open(os.path.join(data_dir, 'label_map.txt'), 'r') as fp:
    label_map = fp.read()
  with open(os.path.join(data_dir, 'label_map.pickle'),'wb' ) as fp:
    pickle.dump(label_map, fp)
  model._load_ckpt(data_dir)
  model.train_with_sgd(train_and_val_dataset, num_iter, learning_rate, batch_size=batch_size, print_every=10)
  t2 = time.perf_counter()
  print('Training takes %.2f seconds' % (t2 - t1))

  # Append learned weights to input model part B and save as tflite format.
  out_model_path = os.path.join(output_dir, 'retrained_model_edgetpu.tflite')
  model.save_as_tflite_model(model_B_path, out_model_path)
  print('Model %s saved.' % out_model_path)
  os.system('mv {0} {1}'.format(os.path.join(data_dir, 'label_map.txt'), os.path.join(output_dir, 'label_map.txt')))
  print('Label map %s saved' % os.path.join(output_dir, 'label_map.txt'))
  t3 = time.perf_counter()
  print('Saving retrained model and label map takes %.2f seconds' % (t3 - t2))
  model._save_ckpt(output_dir)

  # Test
  retrained_engine = basic_engine.BasicEngine(out_model_path)
  test_embeddings = extract_final_embeddings(test_dataset['data_test'], retrained_engine, scaling_factor, zero_point)
  saved_model_acc = np.mean(np.argmax(test_embeddings, axis=1) == test_dataset['labels_test'])
  print(np.argmax(test_embeddings, axis=1))
  print(test_dataset['labels_test'])
  """
  check=[[0]*10 for i in range(10)]
  for i in range(10):
    recognized=False
    x = np.argmax(test_embeddings, axis=1)
    y = test_dataset['labels_test']
    for j in range(199):
      if x[j]==y[j]:
        recognized=True
        if recognized and x[j]==i:
          check[i][i] = check[i][i] + 1
      elif x[j]==i:
        check[y[j]][i] = check[y[j]][i] + 1
  print('--------------------')
  for i in range(10):
    print(check[i])
  """
  print('Saved tflite model test accuracy: %.2f%%' % (saved_model_acc * 100))
  t4 = time.perf_counter()
  print('Checking test accuracy takes %.2f seconds' % (t4 - t3))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--embedding_extractor_B_path', required=True, help='Path to embedding extractor part B tflite model which was compiled.')
  parser.add_argument('--dataset_dir', required=True, help='Directory to the output from edge device 1.')
  parser.add_argument('--output_dir', default='./output', help='Path to directory to save retrained model and label map.')
  parser.add_argument('--scaling_factor', required=True, type=float, help='Scaling factor to quantize the input of part B.')
  parser.add_argument('--zero_point', required=True, type=int, help='Zero point to quantize the input of part B.')
  
  args = parser.parse_args()
  if not os.path.exists(args.dataset_dir):
    sys.exit('%s does not exist!' % args.dataset_dir)
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  part_B_run_and_train(args.embedding_extractor_B_path, args.dataset_dir, args.output_dir, args.scaling_factor, args.zero_point)

if __name__ == '__main__':
  main()
