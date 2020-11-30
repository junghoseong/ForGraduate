"""
Pretrain되어 있는 MobileNetV2를 keras model으로서 불러오고, 이를 .h5 file으로 저장한다.
그 후, 해당 keras model에 integer-only inference를 위한 post-training quantization을 적용하고, 이를 .tflite file으로 저장한다. 
"""
import tensorflow as tf
import numpy as np
import os
import cv2
import argparse

resize_image_first_call = True

def resize_image(image, desired_size, pad_color=0):
	h, w = image.shape[:2]
	desired_h, desired_w = desired_size, desired_size

	# interpolation method
	if h > desired_h or w > desired_w: # shrinking image
	    interp = cv2.INTER_AREA
	else: # stretching image
	    interp = cv2.INTER_CUBIC

	# aspect ratio of image
	aspect = float(w) / h 
	desired_aspect = float(desired_w) / desired_h

	if (desired_aspect > aspect) or ((desired_aspect == 1) and (aspect <= 1)):  # new horizontal image
	    new_h = desired_h
	    new_w = np.round(new_h * aspect).astype(int)
	    pad_horz = float(desired_w - new_w) / 2
	    pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
	    pad_top, pad_bot = 0, 0

	elif (desired_aspect < aspect) or ((desired_aspect == 1) and (aspect >= 1)):  # new vertical image
	    new_w = desired_w
	    new_h = np.round(float(new_w) / aspect).astype(int)
	    pad_vert = float(desired_h - new_h) / 2
	    pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
	    pad_left, pad_right = 0, 0

	# set pad color
	if len(image.shape) is 3 and not isinstance(pad_color, (list, tuple, np.ndarray)): # color image but only one color provided
	    pad_color = [pad_color]*3

	# scale and pad
	scaled_image = cv2.resize(image, (new_w, new_h), interpolation=interp)
	scaled_image = cv2.copyMakeBorder(scaled_image, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=pad_color)

	return scaled_image

def representative_images_gen(imageset_dir, num_samples, desired_size):
	"""
	imageset_dir에서 num_samples만큼의 image data를 sample하여 preprocess한 뒤 
	이들의 list를 representative_images로 출력한다.
	"""
	representative_images = []
	image_paths = os.listdir(imageset_dir)
	for i in range(num_samples):
		image_path = image_paths[i]
		image = cv2.imread(os.path.join(imageset_dir, image_path))
		resized_image = resize_image(image, desired_size)
		normalized_image = resized_image.astype(np.float32) / 255.0
		representative_image = np.reshape(normalized_image, (1,desired_size,desired_size,3))
		representative_images.append(representative_image)
		if i == 0 or (i + 1) % 1000 == 0:
			print('%d번째 representative image processing 완료'%(i+1))
	return representative_images

def main():
	# Argument parsing
	parser = argparse.ArgumentParser()
	parser.add_argument('--imageset_dir', required=True, help='Path to an imageset for representative image(s)')
	parser.add_argument('--num_samples', default=5000, help='Number of samples in a representative imageset')
	parser.add_argument('--desired_size', default=224, help='Desired input image size. In specific, an image is resized to (desired_size) X (desired_size)')
	parser.add_argument('--output_dir', default='./output', help='Path to save outputs')
	args = parser.parse_args()
	if not os.path.exists(args.imageset_dir):
		sys.exit('%s does not exist!' % args.imageset_dir)
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Pretrained MobileNetV2를 불러오고 그 개요를 출력
	# MobileNetV2의 classifier 부분은 불러오지 않는다. (i.e., include_top=False)
	full_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, pooling='avg')
	print('Successfully imported full model')
	full_model.summary()

	# Full model을 .h5 file으로 저장
	MobileNetV2_with_imagenet_full_path = os.path.join(args.output_dir, 'MobileNetV2_with_ImageNet_full.h5') 
	tf.keras.models.save_model(full_model, MobileNetV2_with_imagenet_full_path)
	print('Successfully saved full model to {}'.format(MobileNetV2_with_imagenet_full_path))

	# Full model에 integer-only inference를 위한 post-training quantization을 적용
	representative_images = representative_images_gen(args.imageset_dir, args.num_samples, args.desired_size)
	def representative_data_gen():
		for representative_image in representative_images:
			yield [representative_image]
			
	converter = tf.lite.TFLiteConverter.from_keras_model(full_model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.representative_dataset = representative_data_gen
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter.inference_input_type = tf.uint8
	converter.inference_output_type = tf.uint8
	quant_model = converter.convert()
	print('Successfully generated quantized model')

	# Quantized model을 .tflite file으로 저장
	MobileNetV2_quant_path = os.path.join(args.output_dir, 'MobileNetV2_with_ImageNet_quant.tflite')
	open(MobileNetV2_quant_path, 'wb').write(quant_model)	
	print('Successfully saved the quantized model to {}'.format(MobileNetV2_quant_path))

	# 사전에 edgetpu-compiler를 install하여야 함
	os.system('edgetpu_compiler {0}'.format(MobileNetV2_quant_path))
	print('Successfully compiled the quantized model for inference with edge TPU')

if __name__ == '__main__':
  main()