"""
Pretrain되어 있는 MobileNetV2를 keras model으로서 불러오고, 이를 .h5 file으로 저장한다.
그 후, 해당 keras model에 integer-only inference를 위한 post-training quantization을 적용하고, 이를 .tflite file으로 저장한다. 
"""
import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import argparse
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, \
	ReLU, DepthwiseConv2D, Add, GlobalAveragePooling2D, Dense

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

def define_MobileNetV2_for_A(split_boundary, in_tensor):
	"""
	Define MobileNetV2 model architecture
	"""
	layers = []

	input_1 = tf.keras.Input(tensor=in_tensor)
	layers.append(input_1)
	Conv1_pad = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(input_1)
	layers.append(Conv1_pad)
	Conv1 = Conv2D(filters=32,
		kernel_size=(3,3),
		strides=(2,2),
		data_format='channels_last',
		activation='linear',
		use_bias=False)(Conv1_pad)
	layers.append(Conv1)	
	bn_Conv1 = BatchNormalization(axis=3,
		momentum=0.999)(Conv1)
	layers.append(bn_Conv1)
	Conv1_relu = ReLU(max_value=6)(bn_Conv1)
	layers.append(Conv1_relu)
	if split_boundary == 'expanded_conv_depthwise':
		return layers
	expanded_conv_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(Conv1_relu)
	layers.append(expanded_conv_depthwise)
	expanded_conv_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(expanded_conv_depthwise)
	layers.append(expanded_conv_depthwise_BN)
	expanded_conv_depthwise_relu = ReLU(max_value=6)(expanded_conv_depthwise_BN)
	layers.append(expanded_conv_depthwise_relu)
	if split_boundary == 'expanded_conv_project':
		return layers	
	expanded_conv_project = Conv2D(filters=16,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(expanded_conv_depthwise_relu)
	layers.append(expanded_conv_project)
	expanded_conv_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(expanded_conv_project)
	layers.append(expanded_conv_project_BN)
	if split_boundary == 'block_1_expand':
		return layers
	block_1_expand = Conv2D(filters=96,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(expanded_conv_project_BN)
	layers.append(block_1_expand)
	block_1_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_1_expand)
	layers.append(block_1_expand_BN)
	block_1_expand_relu = ReLU(max_value=6)(block_1_expand_BN)
	layers.append(block_1_expand_relu)
	block_1_pad = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(block_1_expand_relu)
	layers.append(block_1_pad)
	if split_boundary == 'block_1_depthwise':
		return layers
	block_1_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		strides=(2,2),
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_1_pad)
	layers.append(block_1_depthwise)
	block_1_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_1_depthwise)
	layers.append(block_1_depthwise_BN)
	block_1_depthwise_relu = ReLU(max_value=6)(block_1_depthwise_BN)
	layers.append(block_1_depthwise_relu)
	if split_boundary == 'block_1_project':
		return layers
	block_1_project = Conv2D(filters=24,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_1_depthwise_relu)
	layers.append(block_1_project)
	block_1_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_1_project)
	layers.append(block_1_project_BN)
	block_2_expand = Conv2D(filters=144,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_1_project_BN)
	layers.append(block_2_expand)
	block_2_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_2_expand)
	layers.append(block_2_expand_BN)
	block_2_expand_relu = ReLU(max_value=6)(block_2_expand_BN)
	layers.append(block_2_expand_relu)
	block_2_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_2_expand_relu)
	layers.append(block_2_depthwise)
	block_2_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_2_depthwise)
	layers.append(block_2_depthwise_BN)
	block_2_depthwise_relu = ReLU(max_value=6)(block_2_depthwise_BN)
	layers.append(block_2_depthwise_relu)
	block_2_project = Conv2D(filters=24,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_2_depthwise_relu)
	layers.append(block_2_project)
	block_2_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_2_project)
	layers.append(block_2_project_BN)
	block_2_add = Add()([block_1_project_BN, block_2_project_BN])
	layers.append(block_2_add)
	if split_boundary == 'block_3_expand':
		return layers
	block_3_expand = Conv2D(filters=144,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_2_add)
	layers.append(block_3_expand)
	block_3_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_3_expand)
	layers.append(block_3_expand_BN)
	block_3_expand_relu = ReLU(max_value=6)(block_3_expand_BN)
	layers.append(block_3_expand_relu)
	block_3_pad = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(block_3_expand_relu)
	layers.append(block_3_pad)
	if split_boundary == 'block_3_depthwise':
		return layers
	block_3_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		strides=(2,2),
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_3_pad)
	layers.append(block_3_depthwise)
	block_3_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_3_depthwise)
	layers.append(block_3_depthwise_BN)
	block_3_depthwise_relu = ReLU(max_value=6)(block_3_depthwise_BN)
	layers.append(block_3_depthwise_relu)
	if split_boundary == 'block_3_project':
		return layers
	block_3_project = Conv2D(filters=32,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_3_depthwise_relu)
	layers.append(block_3_project)
	block_3_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_3_project)
	layers.append(block_3_project_BN)
	block_4_expand = Conv2D(filters=192,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_3_project_BN)
	layers.append(block_4_expand)
	block_4_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_4_expand)
	layers.append(block_4_expand_BN)
	block_4_expand_relu = ReLU(max_value=6)(block_4_expand_BN)
	layers.append(block_4_expand_relu)
	block_4_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_4_expand_relu)
	layers.append(block_4_depthwise)
	block_4_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_4_depthwise)
	layers.append(block_4_depthwise_BN)
	block_4_depthwise_relu = ReLU(max_value=6)(block_4_depthwise_BN)
	layers.append(block_4_depthwise_relu)
	block_4_project = Conv2D(filters=32,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_4_depthwise_relu)
	layers.append(block_4_project)
	block_4_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_4_project)
	layers.append(block_4_project_BN)
	block_4_add = Add()([block_3_project_BN, block_4_project_BN])
	layers.append(block_4_add)
	block_5_expand = Conv2D(filters=192,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_4_add)
	layers.append(block_5_expand)
	block_5_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_5_expand)
	layers.append(block_5_expand_BN)
	block_5_expand_relu = ReLU(max_value=6)(block_5_expand_BN)
	layers.append(block_5_expand_relu)
	block_5_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_5_expand_relu)
	layers.append(block_5_depthwise)
	block_5_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_5_depthwise)
	layers.append(block_5_depthwise_BN)
	block_5_depthwise_relu = ReLU(max_value=6)(block_5_depthwise_BN)
	layers.append(block_5_depthwise_relu)
	block_5_project = Conv2D(filters=32,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_5_depthwise_relu)
	layers.append(block_5_project)
	block_5_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_5_project)
	layers.append(block_5_project_BN)
	block_5_add = Add()([block_4_add, block_5_project_BN])
	layers.append(block_5_add)
	if split_boundary == 'block_6_expand':
		return layers
	block_6_expand = Conv2D(filters=192,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_5_add)
	layers.append(block_6_expand)
	block_6_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_6_expand)
	layers.append(block_6_expand_BN)
	block_6_expand_relu = ReLU(max_value=6)(block_6_expand_BN)
	layers.append(block_6_expand_relu)
	block_6_pad = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(block_6_expand_relu)
	layers.append(block_6_pad)
	if split_boundary == 'block_6_depthwise':
		return layers
	block_6_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		strides=(2,2),
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_6_pad)
	layers.append(block_6_depthwise)
	block_6_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_6_depthwise)
	layers.append(block_6_depthwise_BN)
	block_6_depthwise_relu = ReLU(max_value=6)(block_6_depthwise_BN)
	layers.append(block_6_depthwise_relu)
	if split_boundary == 'block_6_project':
		return layers
	block_6_project = Conv2D(filters=64,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_6_depthwise_relu)
	layers.append(block_6_project)
	block_6_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_6_project)
	layers.append(block_6_project_BN)
	block_7_expand = Conv2D(filters=384,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_6_project_BN)
	layers.append(block_7_expand)
	block_7_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_7_expand)
	layers.append(block_7_expand_BN)
	block_7_expand_relu = ReLU(max_value=6)(block_7_expand_BN)
	layers.append(block_7_expand_relu)
	block_7_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_7_expand_relu)
	layers.append(block_7_depthwise)
	block_7_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_7_depthwise)
	layers.append(block_7_depthwise_BN)
	block_7_depthwise_relu = ReLU(max_value=6)(block_7_depthwise_BN)
	layers.append(block_7_depthwise_relu)
	block_7_project = Conv2D(filters=64,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_7_depthwise_relu)
	layers.append(block_7_project)
	block_7_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_7_project)
	layers.append(block_7_project_BN)
	block_7_add = Add()([block_6_project_BN, block_7_project_BN])
	layers.append(block_7_add)
	block_8_expand = Conv2D(filters=384,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_7_add)
	layers.append(block_8_expand)
	block_8_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_8_expand)
	layers.append(block_8_expand_BN)
	block_8_expand_relu = ReLU(max_value=6)(block_8_expand_BN)
	layers.append(block_8_expand_relu)
	block_8_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_8_expand_relu)
	layers.append(block_8_depthwise)
	block_8_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_8_depthwise)
	layers.append(block_8_depthwise_BN)
	block_8_depthwise_relu = ReLU(max_value=6)(block_8_depthwise_BN)
	layers.append(block_8_depthwise_relu)
	block_8_project = Conv2D(filters=64,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_8_depthwise_relu)
	layers.append(block_8_project)
	block_8_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_8_project)
	layers.append(block_8_project_BN)
	block_8_add = Add()([block_7_add, block_8_project_BN])
	layers.append(block_8_add)
	block_9_expand = Conv2D(filters=384,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_8_add)
	layers.append(block_9_expand)
	block_9_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_9_expand)
	layers.append(block_9_expand_BN)
	block_9_expand_relu = ReLU(max_value=6)(block_9_expand_BN)
	layers.append(block_9_expand_relu)
	block_9_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_9_expand_relu)
	layers.append(block_9_depthwise)
	block_9_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_9_depthwise)
	layers.append(block_9_depthwise_BN)
	block_9_depthwise_relu = ReLU(max_value=6)(block_9_depthwise_BN)
	layers.append(block_9_depthwise_relu)
	block_9_project = Conv2D(filters=64,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_9_depthwise_relu)
	layers.append(block_9_project)
	block_9_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_9_project)
	layers.append(block_9_project_BN)
	block_9_add = Add()([block_8_add, block_9_project_BN])
	layers.append(block_9_add)
	if split_boundary == 'block_10_expand':
		return layers
	block_10_expand = Conv2D(filters=384,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_9_add)
	layers.append(block_10_expand)
	block_10_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_10_expand)
	layers.append(block_10_expand_BN)
	block_10_expand_relu = ReLU(max_value=6)(block_10_expand_BN)
	layers.append(block_10_expand_relu)
	if split_boundary == 'block_10_depthwise':
		return layers
	block_10_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_10_expand_relu)
	layers.append(block_10_depthwise)
	block_10_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_10_depthwise)
	layers.append(block_10_depthwise_BN)
	block_10_depthwise_relu = ReLU(max_value=6)(block_10_depthwise_BN)
	layers.append(block_10_depthwise_relu)
	if split_boundary == 'block_10_project':
		return layers
	block_10_project = Conv2D(filters=96,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_10_depthwise_relu)
	layers.append(block_10_project)
	block_10_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_10_project)
	layers.append(block_10_project_BN)
	block_11_expand = Conv2D(filters=576,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_10_project_BN)
	layers.append(block_11_expand)
	block_11_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_11_expand)
	layers.append(block_11_expand_BN)
	block_11_expand_relu = ReLU(max_value=6)(block_11_expand_BN)
	layers.append(block_11_expand_relu)
	block_11_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_11_expand_relu)
	layers.append(block_11_depthwise)
	block_11_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_11_depthwise)
	layers.append(block_11_depthwise_BN)
	block_11_depthwise_relu = ReLU(max_value=6)(block_11_depthwise_BN)
	layers.append(block_11_depthwise_relu)
	block_11_project = Conv2D(filters=96,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_11_depthwise_relu)
	layers.append(block_11_project)
	block_11_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_11_project)
	layers.append(block_11_project_BN)
	block_11_add = Add()([block_10_project_BN, block_11_project_BN])
	layers.append(block_11_add)
	block_12_expand = Conv2D(filters=576,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_11_add)
	layers.append(block_12_expand)
	block_12_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_12_expand)
	layers.append(block_12_expand_BN)
	block_12_expand_relu = ReLU(max_value=6)(block_12_expand_BN)
	layers.append(block_12_expand_relu)
	block_12_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_12_expand_relu)
	layers.append(block_12_depthwise)
	block_12_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_12_depthwise)
	layers.append(block_12_depthwise_BN)
	block_12_depthwise_relu = ReLU(max_value=6)(block_12_depthwise_BN)
	layers.append(block_12_depthwise_relu)
	block_12_project = Conv2D(filters=96,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_12_depthwise_relu)
	layers.append(block_12_project)
	block_12_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_12_project)
	layers.append(block_12_project_BN)
	block_12_add = Add()([block_11_add, block_12_project_BN])
	layers.append(block_12_add)
	if split_boundary == 'block_13_expand':
		return layers
	block_13_expand = Conv2D(filters=576,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_12_add)
	layers.append(block_13_expand)
	block_13_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_13_expand)
	layers.append(block_13_expand_BN)
	block_13_expand_relu = ReLU(max_value=6)(block_13_expand_BN)
	layers.append(block_13_expand_relu)
	block_13_pad = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(block_13_expand_relu)
	layers.append(block_13_pad)
	if split_boundary == 'block_13_depthwise':
		return layers
	block_13_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		strides=(2,2),
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_13_pad)
	layers.append(block_13_depthwise)
	block_13_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_13_depthwise)
	layers.append(block_13_depthwise_BN)
	block_13_depthwise_relu = ReLU(max_value=6)(block_13_depthwise_BN)
	layers.append(block_13_depthwise_relu)
	if split_boundary == 'block_13_project':
		return layers
	block_13_project = Conv2D(filters=160,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_13_depthwise_relu)
	layers.append(block_13_project)
	block_13_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_13_project)
	layers.append(block_13_project_BN)
	block_14_expand = Conv2D(filters=960,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_13_project_BN)
	layers.append(block_14_expand)
	block_14_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_14_expand)
	layers.append(block_14_expand_BN)
	block_14_expand_relu = ReLU(max_value=6)(block_14_expand_BN)
	layers.append(block_14_expand_relu)
	block_14_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_14_expand_relu)
	layers.append(block_14_depthwise)
	block_14_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_14_depthwise)
	layers.append(block_14_depthwise_BN)
	block_14_depthwise_relu = ReLU(max_value=6)(block_14_depthwise_BN)
	layers.append(block_14_depthwise_relu)
	block_14_project = Conv2D(filters=160,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_14_depthwise_relu)
	layers.append(block_14_project)
	block_14_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_14_project)
	layers.append(block_14_project_BN)
	block_14_add = Add()([block_13_project_BN, block_14_project_BN])
	layers.append(block_14_add)
	block_15_expand = Conv2D(filters=960,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_14_add)
	layers.append(block_15_expand)
	block_15_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_15_expand)
	layers.append(block_15_expand_BN)
	block_15_expand_relu = ReLU(max_value=6)(block_15_expand_BN)
	layers.append(block_15_expand_relu)
	block_15_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_15_expand_relu)
	layers.append(block_15_depthwise)
	block_15_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_15_depthwise)
	layers.append(block_15_depthwise_BN)
	block_15_depthwise_relu = ReLU(max_value=6)(block_15_depthwise_BN)
	layers.append(block_15_depthwise_relu)
	block_15_project = Conv2D(filters=160,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_15_depthwise_relu)
	layers.append(block_15_project)
	block_15_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_15_project)
	layers.append(block_15_project_BN)
	block_15_add = Add()([block_14_add, block_15_project_BN])
	layers.append(block_15_add)
	if split_boundary == 'block_16_expand':
		return layers
	block_16_expand = Conv2D(filters=960,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_15_add)
	layers.append(block_16_expand)
	block_16_expand_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_16_expand)
	layers.append(block_16_expand_BN)
	block_16_expand_relu = ReLU(max_value=6)(block_16_expand_BN)
	layers.append(block_16_expand_relu)
	if split_boundary == 'block_16_depthwise':
		return layers
	block_16_depthwise = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_16_expand_relu)
	layers.append(block_16_depthwise)
	block_16_depthwise_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_16_depthwise)
	layers.append(block_16_depthwise_BN)
	block_16_depthwise_relu = ReLU(max_value=6)(block_16_depthwise_BN)
	layers.append(block_16_depthwise_relu)
	if split_boundary == 'block_16_project':
		return layers
	block_16_project = Conv2D(filters=320,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_16_depthwise_relu)
	layers.append(block_16_project)
	block_16_project_BN = BatchNormalization(axis=3,
		momentum=0.999)(block_16_project)
	layers.append(block_16_project_BN)
	if split_boundary == 'Conv_1':
		return layers
	Conv_1 = Conv2D(filters=1280,
		kernel_size=(1,1),
		data_format='channels_last',
		activation='linear',
		use_bias=False)(block_16_project_BN)
	layers.append(Conv_1)
	Conv_1_bn = BatchNormalization(axis=3,
		momentum=0.999)(Conv_1)
	layers.append(Conv_1_bn)
	out_relu = ReLU(max_value=6)(Conv_1_bn)
	layers.append(out_relu)
	global_average_pooling2d = GlobalAveragePooling2D(data_format='channels_last')(out_relu)
	layers.append(global_average_pooling2d)
	# pred_1 = Dense(units=1000, activation='softmax')(global_average_pooling2d)
	# entire_model = tf.keras.Model(inputs=input_1, outputs=pred_1)
	return layers

def define_MobileNetV2_for_B(split_boundary, in_tensor):
	"""
	Define MobileNetV2 model architecture
	"""
	layers = []
	split_boundary_candidates = ['expanded_conv_depthwise',
		'expanded_conv_project', 'block_1_expand', 'block_1_depthwise',
		'block_1_project', 'block_3_expand', 'block_3_depthwise',
		'block_3_project', 'block_6_expand', 'block_6_depthwise',
		'block_6_project', 'block_10_expand', 'block_10_depthwise',
		'block_10_project', 'block_13_expand', 'block_13_depthwise',
		'block_13_project', 'block_16_expand', 'block_16_depthwise',
		'block_16_project', 'Conv_1']
	cond = split_boundary_candidates.index(split_boundary)

	input_2 = tf.keras.Input(tensor=in_tensor)

	if cond == 0:
		Conv1_relu = input_2
		layers.append(Conv1_relu)

		expanded_conv_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(Conv1_relu)
		layers.append(expanded_conv_depthwise)
		expanded_conv_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(expanded_conv_depthwise)
		layers.append(expanded_conv_depthwise_BN)
		expanded_conv_depthwise_relu = ReLU(max_value=6)(expanded_conv_depthwise_BN)
		layers.append(expanded_conv_depthwise_relu)

	if cond == 1:
		expanded_conv_depthwise_relu = input_2
		layers.append(expanded_conv_depthwise_relu)
	if cond in range(2):
		expanded_conv_project = Conv2D(filters=16,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(expanded_conv_depthwise_relu)
		layers.append(expanded_conv_project)
		expanded_conv_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(expanded_conv_project)
		layers.append(expanded_conv_project_BN)
	
	if cond == 2:
		expanded_conv_project_BN = input_2
		layers.append(expanded_conv_project_BN)
	if cond in range(3):
		block_1_expand = Conv2D(filters=96,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(expanded_conv_project_BN)
		layers.append(block_1_expand)
		block_1_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_1_expand)
		layers.append(block_1_expand_BN)
		block_1_expand_relu = ReLU(max_value=6)(block_1_expand_BN)
		layers.append(block_1_expand_relu)
		block_1_pad = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(block_1_expand_relu)
		layers.append(block_1_pad)

	if cond == 3:
		block_1_pad = input_2
		layers.append(block_1_pad)
	if cond in range(4):
		block_1_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			strides=(2,2),
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_1_pad)
		layers.append(block_1_depthwise)
		block_1_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_1_depthwise)
		layers.append(block_1_depthwise_BN)
		block_1_depthwise_relu = ReLU(max_value=6)(block_1_depthwise_BN)
		layers.append(block_1_depthwise_relu)
	
	if cond == 4:
		block_1_depthwise_relu = input_2
		layers.append(block_1_depthwise_relu)
	if cond in range(5):
		block_1_project = Conv2D(filters=24,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_1_depthwise_relu)
		layers.append(block_1_project)
		block_1_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_1_project)
		layers.append(block_1_project_BN)
		block_2_expand = Conv2D(filters=144,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_1_project_BN)
		layers.append(block_2_expand)
		block_2_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_2_expand)
		layers.append(block_2_expand_BN)
		block_2_expand_relu = ReLU(max_value=6)(block_2_expand_BN)
		layers.append(block_2_expand_relu)
		block_2_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_2_expand_relu)
		layers.append(block_2_depthwise)
		block_2_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_2_depthwise)
		layers.append(block_2_depthwise_BN)
		block_2_depthwise_relu = ReLU(max_value=6)(block_2_depthwise_BN)
		layers.append(block_2_depthwise_relu)
		block_2_project = Conv2D(filters=24,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_2_depthwise_relu)
		layers.append(block_2_project)
		block_2_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_2_project)
		layers.append(block_2_project_BN)
		block_2_add = Add()([block_1_project_BN, block_2_project_BN])
		layers.append(block_2_add)

	if cond == 5:
		block_2_add = input_2
		layers.append(block_2_add)
	if cond in range(6):
		block_3_expand = Conv2D(filters=144,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_2_add)
		layers.append(block_3_expand)
		block_3_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_3_expand)
		layers.append(block_3_expand_BN)
		block_3_expand_relu = ReLU(max_value=6)(block_3_expand_BN)
		layers.append(block_3_expand_relu)
		block_3_pad = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(block_3_expand_relu)
		layers.append(block_3_pad)

	if cond == 6:
		block_3_pad = input_2
		layers.append(block_3_pad)
	if cond in range(7):
		block_3_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			strides=(2,2),
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_3_pad)
		layers.append(block_3_depthwise)
		block_3_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_3_depthwise)
		layers.append(block_3_depthwise_BN)
		block_3_depthwise_relu = ReLU(max_value=6)(block_3_depthwise_BN)
		layers.append(block_3_depthwise_relu)

	if cond == 7:
		block_3_depthwise_relu = input_2
		layers.append(block_3_depthwise_relu)
	if cond in range(8):
		block_3_project = Conv2D(filters=32,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_3_depthwise_relu)
		layers.append(block_3_project)
		block_3_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_3_project)
		layers.append(block_3_project_BN)
		block_4_expand = Conv2D(filters=192,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_3_project_BN)
		layers.append(block_4_expand)
		block_4_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_4_expand)
		layers.append(block_4_expand_BN)
		block_4_expand_relu = ReLU(max_value=6)(block_4_expand_BN)
		layers.append(block_4_expand_relu)
		block_4_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_4_expand_relu)
		layers.append(block_4_depthwise)
		block_4_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_4_depthwise)
		layers.append(block_4_depthwise_BN)
		block_4_depthwise_relu = ReLU(max_value=6)(block_4_depthwise_BN)
		layers.append(block_4_depthwise_relu)
		block_4_project = Conv2D(filters=32,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_4_depthwise_relu)
		layers.append(block_4_project)
		block_4_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_4_project)
		layers.append(block_4_project_BN)
		block_4_add = Add()([block_3_project_BN, block_4_project_BN])
		layers.append(block_4_add)
		block_5_expand = Conv2D(filters=192,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_4_add)
		layers.append(block_5_expand)
		block_5_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_5_expand)
		layers.append(block_5_expand_BN)
		block_5_expand_relu = ReLU(max_value=6)(block_5_expand_BN)
		layers.append(block_5_expand_relu)
		block_5_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_5_expand_relu)
		layers.append(block_5_depthwise)
		block_5_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_5_depthwise)
		layers.append(block_5_depthwise_BN)
		block_5_depthwise_relu = ReLU(max_value=6)(block_5_depthwise_BN)
		layers.append(block_5_depthwise_relu)
		block_5_project = Conv2D(filters=32,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_5_depthwise_relu)
		layers.append(block_5_project)
		block_5_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_5_project)
		layers.append(block_5_project_BN)
		block_5_add = Add()([block_4_add, block_5_project_BN])
		layers.append(block_5_add)
	
	if cond == 8:
		block_5_add = input_2
		layers.append(block_5_add)
	if cond in range(9):
		block_6_expand = Conv2D(filters=192,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_5_add)
		layers.append(block_6_expand)
		block_6_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_6_expand)
		layers.append(block_6_expand_BN)
		block_6_expand_relu = ReLU(max_value=6)(block_6_expand_BN)
		layers.append(block_6_expand_relu)
		block_6_pad = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(block_6_expand_relu)
		layers.append(block_6_pad)

	if cond == 9:
		block_6_pad = input_2
		layers.append(block_6_pad)
	if cond in range(10):
		block_6_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			strides=(2,2),
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_6_pad)
		layers.append(block_6_depthwise)
		block_6_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_6_depthwise)
		layers.append(block_6_depthwise_BN)
		block_6_depthwise_relu = ReLU(max_value=6)(block_6_depthwise_BN)
		layers.append(block_6_depthwise_relu)

	if cond == 10:
		block_6_depthwise_relu = input_2
		layers.append(block_6_depthwise_relu)
	if cond in range(11):
		block_6_project = Conv2D(filters=64,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_6_depthwise_relu)
		layers.append(block_6_project)
		block_6_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_6_project)
		layers.append(block_6_project_BN)
		block_7_expand = Conv2D(filters=384,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_6_project_BN)
		layers.append(block_7_expand)
		block_7_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_7_expand)
		layers.append(block_7_expand_BN)
		block_7_expand_relu = ReLU(max_value=6)(block_7_expand_BN)
		layers.append(block_7_expand_relu)
		block_7_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_7_expand_relu)
		layers.append(block_7_depthwise)
		block_7_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_7_depthwise)
		layers.append(block_7_depthwise_BN)
		block_7_depthwise_relu = ReLU(max_value=6)(block_7_depthwise_BN)
		layers.append(block_7_depthwise_relu)
		block_7_project = Conv2D(filters=64,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_7_depthwise_relu)
		layers.append(block_7_project)
		block_7_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_7_project)
		layers.append(block_7_project_BN)
		block_7_add = Add()([block_6_project_BN, block_7_project_BN])
		layers.append(block_7_add)
		block_8_expand = Conv2D(filters=384,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_7_add)
		layers.append(block_8_expand)
		block_8_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_8_expand)
		layers.append(block_8_expand_BN)
		block_8_expand_relu = ReLU(max_value=6)(block_8_expand_BN)
		layers.append(block_8_expand_relu)
		block_8_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_8_expand_relu)
		layers.append(block_8_depthwise)
		block_8_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_8_depthwise)
		layers.append(block_8_depthwise_BN)
		block_8_depthwise_relu = ReLU(max_value=6)(block_8_depthwise_BN)
		layers.append(block_8_depthwise_relu)
		block_8_project = Conv2D(filters=64,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_8_depthwise_relu)
		layers.append(block_8_project)
		block_8_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_8_project)
		layers.append(block_8_project_BN)
		block_8_add = Add()([block_7_add, block_8_project_BN])
		layers.append(block_8_add)
		block_9_expand = Conv2D(filters=384,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_8_add)
		layers.append(block_9_expand)
		block_9_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_9_expand)
		layers.append(block_9_expand_BN)
		block_9_expand_relu = ReLU(max_value=6)(block_9_expand_BN)
		layers.append(block_9_expand_relu)
		block_9_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_9_expand_relu)
		layers.append(block_9_depthwise)
		block_9_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_9_depthwise)
		layers.append(block_9_depthwise_BN)
		block_9_depthwise_relu = ReLU(max_value=6)(block_9_depthwise_BN)
		layers.append(block_9_depthwise_relu)
		block_9_project = Conv2D(filters=64,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_9_depthwise_relu)
		layers.append(block_9_project)
		block_9_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_9_project)
		layers.append(block_9_project_BN)
		block_9_add = Add()([block_8_add, block_9_project_BN])
		layers.append(block_9_add)

	if cond == 11:
		block_9_add = input_2
		layers.append(block_9_add)
	if cond in range(12):
		block_10_expand = Conv2D(filters=384,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_9_add)
		layers.append(block_10_expand)
		block_10_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_10_expand)
		layers.append(block_10_expand_BN)
		block_10_expand_relu = ReLU(max_value=6)(block_10_expand_BN)
		layers.append(block_10_expand_relu)

	if cond == 12:
		block_10_expand_relu = input_2
		layers.append(block_10_expand_relu)
	if cond in range(13):
		block_10_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_10_expand_relu)
		layers.append(block_10_depthwise)
		block_10_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_10_depthwise)
		layers.append(block_10_depthwise_BN)
		block_10_depthwise_relu = ReLU(max_value=6)(block_10_depthwise_BN)
		layers.append(block_10_depthwise_relu)

	if cond == 13:
		block_10_depthwise_relu = input_2
		layers.append(block_10_depthwise_relu)
	if cond in range(14):
		block_10_project = Conv2D(filters=96,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_10_depthwise_relu)
		layers.append(block_10_project)
		block_10_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_10_project)
		layers.append(block_10_project_BN)
		block_11_expand = Conv2D(filters=576,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_10_project_BN)
		layers.append(block_11_expand)
		block_11_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_11_expand)
		layers.append(block_11_expand_BN)
		block_11_expand_relu = ReLU(max_value=6)(block_11_expand_BN)
		layers.append(block_11_expand_relu)
		block_11_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_11_expand_relu)
		layers.append(block_11_depthwise)
		block_11_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_11_depthwise)
		layers.append(block_11_depthwise_BN)
		block_11_depthwise_relu = ReLU(max_value=6)(block_11_depthwise_BN)
		layers.append(block_11_depthwise_relu)
		block_11_project = Conv2D(filters=96,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_11_depthwise_relu)
		layers.append(block_11_project)
		block_11_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_11_project)
		layers.append(block_11_project_BN)
		block_11_add = Add()([block_10_project_BN, block_11_project_BN])
		layers.append(block_11_add)
		block_12_expand = Conv2D(filters=576,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_11_add)
		layers.append(block_12_expand)
		block_12_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_12_expand)
		layers.append(block_12_expand_BN)
		block_12_expand_relu = ReLU(max_value=6)(block_12_expand_BN)
		layers.append(block_12_expand_relu)
		block_12_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_12_expand_relu)
		layers.append(block_12_depthwise)
		block_12_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_12_depthwise)
		layers.append(block_12_depthwise_BN)
		block_12_depthwise_relu = ReLU(max_value=6)(block_12_depthwise_BN)
		layers.append(block_12_depthwise_relu)
		block_12_project = Conv2D(filters=96,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_12_depthwise_relu)
		layers.append(block_12_project)
		block_12_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_12_project)
		layers.append(block_12_project_BN)
		block_12_add = Add()([block_11_add, block_12_project_BN])
		layers.append(block_12_add)

	if cond == 14:
		block_12_add = input_2
		layers.append(block_12_add)
	if cond in range(15):
		block_13_expand = Conv2D(filters=576,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_12_add)
		layers.append(block_13_expand)
		block_13_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_13_expand)
		layers.append(block_13_expand_BN)
		block_13_expand_relu = ReLU(max_value=6)(block_13_expand_BN)
		layers.append(block_13_expand_relu)
		block_13_pad = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(block_13_expand_relu)
		layers.append(block_13_pad)

	if cond == 15:
		block_13_pad = input_2
		layers.append(block_13_pad)
	if cond in range(16):
		block_13_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			strides=(2,2),
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_13_pad)
		layers.append(block_13_depthwise)
		block_13_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_13_depthwise)
		layers.append(block_13_depthwise_BN)
		block_13_depthwise_relu = ReLU(max_value=6)(block_13_depthwise_BN)
		layers.append(block_13_depthwise_relu)

	if cond == 16:
		block_13_depthwise_relu = input_2
		layers.append(block_13_depthwise_relu)
	if cond in range(17):
		block_13_project = Conv2D(filters=160,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_13_depthwise_relu)
		layers.append(block_13_project)
		block_13_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_13_project)
		layers.append(block_13_project_BN)
		block_14_expand = Conv2D(filters=960,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_13_project_BN)
		layers.append(block_14_expand)
		block_14_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_14_expand)
		layers.append(block_14_expand_BN)
		block_14_expand_relu = ReLU(max_value=6)(block_14_expand_BN)
		layers.append(block_14_expand_relu)
		block_14_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_14_expand_relu)
		layers.append(block_14_depthwise)
		block_14_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_14_depthwise)
		layers.append(block_14_depthwise_BN)
		block_14_depthwise_relu = ReLU(max_value=6)(block_14_depthwise_BN)
		layers.append(block_14_depthwise_relu)
		block_14_project = Conv2D(filters=160,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_14_depthwise_relu)
		layers.append(block_14_project)
		block_14_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_14_project)
		layers.append(block_14_project_BN)
		block_14_add = Add()([block_13_project_BN, block_14_project_BN])
		layers.append(block_14_add)
		block_15_expand = Conv2D(filters=960,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_14_add)
		layers.append(block_15_expand)
		block_15_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_15_expand)
		layers.append(block_15_expand_BN)
		block_15_expand_relu = ReLU(max_value=6)(block_15_expand_BN)
		layers.append(block_15_expand_relu)
		block_15_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_15_expand_relu)
		layers.append(block_15_depthwise)
		block_15_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_15_depthwise)
		layers.append(block_15_depthwise_BN)
		block_15_depthwise_relu = ReLU(max_value=6)(block_15_depthwise_BN)
		layers.append(block_15_depthwise_relu)
		block_15_project = Conv2D(filters=160,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_15_depthwise_relu)
		layers.append(block_15_project)
		block_15_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_15_project)
		layers.append(block_15_project_BN)
		block_15_add = Add()([block_14_add, block_15_project_BN])
		layers.append(block_15_add)

	if cond == 17:
		block_15_add = input_2
		layers.append(block_15_add)
	if cond in range(18):
		block_16_expand = Conv2D(filters=960,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_15_add)
		layers.append(block_16_expand)
		block_16_expand_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_16_expand)
		layers.append(block_16_expand_BN)
		block_16_expand_relu = ReLU(max_value=6)(block_16_expand_BN)
		layers.append(block_16_expand_relu)

	if cond == 18:
		block_16_expand_relu = input_2
		layers.append(block_16_expand_relu)
	if cond in range(19):
		block_16_depthwise = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_16_expand_relu)
		layers.append(block_16_depthwise)
		block_16_depthwise_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_16_depthwise)
		layers.append(block_16_depthwise_BN)
		block_16_depthwise_relu = ReLU(max_value=6)(block_16_depthwise_BN)
		layers.append(block_16_depthwise_relu)

	if cond == 19:
		block_16_depthwise_relu = input_2
		layers.append(block_16_depthwise_relu)
	if cond in range(20):
		block_16_project = Conv2D(filters=320,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_16_depthwise_relu)
		layers.append(block_16_project)
		block_16_project_BN = BatchNormalization(axis=3,
			momentum=0.999)(block_16_project)
		layers.append(block_16_project_BN)

	if cond == 20:
		block_16_project_BN = input_2
		layers.append(block_16_project_BN)
	if cond in range(21):
		Conv_1 = Conv2D(filters=1280,
			kernel_size=(1,1),
			data_format='channels_last',
			activation='linear',
			use_bias=False)(block_16_project_BN)
		layers.append(Conv_1)
		Conv_1_bn = BatchNormalization(axis=3,
			momentum=0.999)(Conv_1)
		layers.append(Conv_1_bn)
		out_relu = ReLU(max_value=6)(Conv_1_bn)
		layers.append(out_relu)
		global_average_pooling2d = GlobalAveragePooling2D(data_format='channels_last')(out_relu)
		layers.append(global_average_pooling2d)
		# pred_1 = Dense(units=1000, activation='softmax')(global_average_pooling2d)
		# entire_model = tf.keras.Model(inputs=input_1, outputs=pred_1)

	return layers	

def split(full_model, split_boundary):
	layer_names = [layer.name for layer in full_model.layers]
	split_boundary_idx = layer_names.index(split_boundary) 

	layers = define_MobileNetV2_for_A(split_boundary, full_model.input)
	full_A_model = tf.keras.Model(inputs=layers[0], outputs=layers[-1])
	for idx in range(1, len(full_A_model.layers)):
		full_A_model.layers[idx].set_weights(full_model.layers[idx].get_weights())
	full_A_model.summary()

	layers = define_MobileNetV2_for_B(split_boundary, full_model.layers[split_boundary_idx].input)
	full_B_model = tf.keras.Model(inputs=layers[0], outputs=layers[-1])
	for idx in range(1, len(full_B_model.layers)):
		full_B_model.layers[idx].set_weights(full_model.layers[len(full_A_model.layers) - 1 + idx].get_weights())
	full_B_model.summary()

	return full_A_model, full_B_model, layer_names, split_boundary_idx

def main():
	# Argument parsing
	parser = argparse.ArgumentParser()
	parser.add_argument('--split_boundary', required=True, help='The name of split boundary layer.')
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

	full_A_model, full_B_model, layer_names, split_boundary_idx = split(full_model, args.split_boundary)
	print('Successfully split full model into part A and B')

	# full_A_model과 full_B_model을 각각 .h5 file으로 저장
	MobileNetV2_with_imagenet_full_A_path = os.path.join(args.output_dir, 'MobileNetV2_with_ImageNet_full_A.h5') 
	MobileNetV2_with_imagenet_full_A_pb_path = os.path.join(args.output_dir, 'MobileNetV2_with_ImageNet_full_A_pb') 
	MobileNetV2_with_imagenet_full_B_path = os.path.join(args.output_dir, 'MobileNetV2_with_ImageNet_full_B.h5') 
	
	full_A_model.save(MobileNetV2_with_imagenet_full_A_pb_path, save_format='tf')
	imported = tf.saved_model.load(MobileNetV2_with_imagenet_full_A_pb_path)

	tf.keras.models.save_model(full_A_model, MobileNetV2_with_imagenet_full_A_path)
	tf.keras.models.save_model(full_B_model, MobileNetV2_with_imagenet_full_B_path)
	print('Successfully saved part A to {}'.format(MobileNetV2_with_imagenet_full_A_path))
	print('Successfully saved part B to {}'.format(MobileNetV2_with_imagenet_full_B_path))

	# 각 model에 integer-only inference를 위한 post-training quantization을 적용
	representative_images = representative_images_gen(args.imageset_dir, args.num_samples, args.desired_size)
	def representative_data_gen_A():
		for representative_image in representative_images:
			yield [representative_image]
	def representative_data_gen_B():
		for representative_image in representative_images:
			x = representative_image
			x = imported(x)
			representative_feature = x.numpy()
			yield [representative_feature]

	converter_A = tf.lite.TFLiteConverter.from_keras_model(full_A_model)
	converter_A.optimizations = [tf.lite.Optimize.DEFAULT]
	converter_A.representative_dataset = representative_data_gen_A
	converter_A.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter_A.inference_input_type = tf.uint8
	converter_A.inference_output_type = tf.uint8
	quant_A_model = converter_A.convert()
	print('Successfully generated quantized part A')

	converter_B = tf.lite.TFLiteConverter.from_keras_model(full_B_model)
	converter_B.optimizations = [tf.lite.Optimize.DEFAULT]
	converter_B.representative_dataset = representative_data_gen_B
	converter_B.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter_B.inference_input_type = tf.uint8
	converter_B.inference_output_type = tf.uint8
	quant_B_model = converter_B.convert()
	print('Successfully generated quantized part B')

	# 각 Quantized model을 .tflite file으로 저장
	MobileNetV2_quant_A_path = os.path.join(args.output_dir, 'MobileNetV2_with_ImageNet_quant_A.tflite')
	MobileNetV2_quant_B_path = os.path.join(args.output_dir, 'MobileNetV2_with_ImageNet_quant_B.tflite')
	open(MobileNetV2_quant_A_path, 'wb').write(quant_A_model)	
	print('Successfully saved the quantized part A to {}'.format(MobileNetV2_quant_A_path))
	open(MobileNetV2_quant_B_path, 'wb').write(quant_B_model)	
	print('Successfully saved the quantized part B to {}'.format(MobileNetV2_quant_B_path))

	# 사전에 edgetpu-compiler를 install하여야 함
	os.system('edgetpu_compiler {0}'.format(MobileNetV2_quant_A_path))
	os.system('edgetpu_compiler {0}'.format(MobileNetV2_quant_B_path))
	print('Successfully compiled the quantized parts for inference with edge TPU')

if __name__ == '__main__':
  main()
