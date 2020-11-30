"""
Pretrain되어 있는 MobileNet를 keras model으로서 불러오고, 이를 .h5 file으로 저장한다.
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

def define_MobileNet_for_A(split_boundary, in_tensor):
	"""
	Define MobileNet model architecture
	"""
	layers = []

	input_1 = tf.keras.Input(tensor=in_tensor)
	layers.append(input_1)

	conv1_pad = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(input_1)
	layers.append(conv1_pad)
	
	conv1 = Conv2D(filters=32,
		kernel_size=(3,3),
		strides=(2,2),
		padding='valid',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv1_pad)
	layers.append(conv1)	
	
	conv1_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv1)
	layers.append(conv1_bn)
	
	conv1_relu = ReLU(max_value=6)(conv1_bn)
	layers.append(conv1_relu)
	
	if split_boundary == 'conv_dw_1':
		return layers
	conv_dw_1 = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv1_relu)
	layers.append(conv_dw_1)
	
	conv_dw_1_bn= BatchNormalization(axis=3,
		momentum=0.99)(conv_dw_1)
	layers.append(conv_dw_1_bn)
	
	conv_dw_1_relu = ReLU(max_value=6)(conv_dw_1_bn)
	layers.append(conv_dw_1_relu)	
	
	conv_pw_1 = Conv2D(filters=64,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_dw_1_relu)
	layers.append(conv_pw_1)
	
	conv_pw_1_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_pw_1)
	layers.append(conv_pw_1_bn)
	
	conv_pw_1_relu = ReLU(max_value=6)(conv_pw_1_bn)
	layers.append(conv_pw_1_relu)
	
	conv_pad_2 = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(conv_pw_1_relu)
	layers.append(conv_pad_2)
	
	if split_boundary == 'conv_dw_2':
		return layers
	conv_dw_2 = DepthwiseConv2D(kernel_size=(3,3),
		padding='valid',
		data_format='channels_last',
		activation='linear',
		strides = (2, 2),
		use_bias=False)(conv_pad_2)
	layers.append(conv_dw_2)
	

	conv_dw_2_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_dw_2)
	layers.append(conv_dw_2_bn)
	
	conv_dw_2_relu = ReLU(max_value=6)(conv_dw_2_bn)
	layers.append(conv_dw_2_relu)
	
	conv_pw_2 = Conv2D(filters=128,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_dw_2_relu)
	layers.append(conv_pw_2)

	conv_pw_2_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_pw_2)
	layers.append(conv_pw_2_bn)

	conv_pw_2_relu = ReLU(max_value=6)(conv_pw_2_bn)
	layers.append(conv_pw_2_relu)

	if split_boundary == 'conv_dw_3':
		return layers
	conv_dw_3 = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_pw_2_relu)
	layers.append(conv_dw_3)
	

	conv_dw_3_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_dw_3)
	layers.append(conv_dw_3_bn)

	conv_dw_3_relu = ReLU(max_value=6)(conv_dw_3_bn)
	layers.append(conv_dw_3_relu)

	conv_pw_3 = Conv2D(filters=128,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_dw_3_relu)
	layers.append(conv_pw_3)

	conv_pw_3_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_pw_3)
	layers.append(conv_pw_3_bn)

	conv_pw_3_relu = ReLU(max_value=6)(conv_pw_3_bn)
	layers.append(conv_pw_3_relu)

	conv_pad_4 = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(conv_pw_3_relu)
	layers.append(conv_pad_4)

	if split_boundary == 'conv_dw_4':
		return layers
	conv_dw_4 = DepthwiseConv2D(kernel_size=(3,3),
		padding='valid',
		data_format='channels_last',
		activation='linear',
		strides = (2, 2),
		use_bias=False)(conv_pad_4)
	layers.append(conv_dw_4)
	
	conv_dw_4_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_dw_4)
	layers.append(conv_dw_4_bn)

	conv_dw_4_relu = ReLU(max_value=6)(conv_dw_4_bn)
	layers.append(conv_dw_4_relu)

	conv_pw_4 = Conv2D(filters=256,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_dw_4_relu)
	layers.append(conv_pw_4)

	conv_pw_4_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_pw_4)
	layers.append(conv_pw_4_bn)

	conv_pw_4_relu = ReLU(max_value=6)(conv_pw_4_bn)
	layers.append(conv_pw_4_relu)

	if split_boundary == 'conv_dw_5':
		return layers
	conv_dw_5 = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_pw_4_relu)
	layers.append(conv_dw_5)

	conv_dw_5_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_dw_5)
	layers.append(conv_dw_5_bn)

	conv_dw_5_relu = ReLU(max_value=6)(conv_dw_5_bn)
	layers.append(conv_dw_5_relu)

	conv_pw_5 = Conv2D(filters=256,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_dw_5_relu)
	layers.append(conv_pw_5)

	conv_pw_5_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_pw_5)
	layers.append(conv_pw_5_bn)

	conv_pw_5_relu = ReLU(max_value=6)(conv_pw_5_bn)
	layers.append(conv_pw_5_relu)

	conv_pad_6 = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(conv_pw_5_relu)
	layers.append(conv_pad_6)

	if split_boundary == 'conv_dw_6':
		return layers
	conv_dw_6 = DepthwiseConv2D(kernel_size=(3,3),
		padding='valid',
		data_format='channels_last',
		activation='linear',
		strides = (2, 2),
		use_bias=False)(conv_pad_6)
	layers.append(conv_dw_6)

	conv_dw_6_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_dw_6)
	layers.append(conv_dw_6_bn)

	conv_dw_6_relu = ReLU(max_value=6)(conv_dw_6_bn)
	layers.append(conv_dw_6_relu)

	conv_pw_6 = Conv2D(filters=512,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_dw_6_relu)
	layers.append(conv_pw_6)

	conv_pw_6_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_pw_6)
	layers.append(conv_pw_6_bn)

	conv_pw_6_relu = ReLU(max_value=6)(conv_pw_6_bn)
	layers.append(conv_pw_6_relu)

	if split_boundary == 'conv_dw_7':
		return layers
	conv_dw_7 = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_pw_6_relu)
	layers.append(conv_dw_7)

	conv_dw_7_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_dw_7)
	layers.append(conv_dw_7_bn)

	conv_dw_7_relu = ReLU(max_value=6)(conv_dw_7_bn)
	layers.append(conv_dw_7_relu)

	conv_pw_7 = Conv2D(filters=512,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_dw_7_relu)
	layers.append(conv_pw_7)

	conv_pw_7_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_pw_7)
	layers.append(conv_pw_7_bn)

	conv_pw_7_relu = ReLU(max_value=6)(conv_pw_7_bn)
	layers.append(conv_pw_7_relu)

	if split_boundary == 'conv_dw_8':
		return layers
	conv_dw_8 = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_pw_7_relu)
	layers.append(conv_dw_8)

	conv_dw_8_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_dw_8)
	layers.append(conv_dw_8_bn)

	conv_dw_8_relu = ReLU(max_value=6)(conv_dw_8_bn)
	layers.append(conv_dw_8_relu)

	conv_pw_8 = Conv2D(filters=512,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_dw_8_relu)
	layers.append(conv_pw_8)

	conv_pw_8_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_pw_8)
	layers.append(conv_pw_8_bn)

	conv_pw_8_relu = ReLU(max_value=6)(conv_pw_8_bn)
	layers.append(conv_pw_8_relu)

	if split_boundary == 'conv_dw_9':
		return layers
	conv_dw_9 = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_pw_8_relu)
	layers.append(conv_dw_9)

	conv_dw_9_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_dw_9)
	layers.append(conv_dw_9_bn)

	conv_dw_9_relu = ReLU(max_value=6)(conv_dw_9_bn)
	layers.append(conv_dw_9_relu)

	conv_pw_9 = Conv2D(filters=512,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_dw_9_relu)
	layers.append(conv_pw_9)

	conv_pw_9_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_pw_9)
	layers.append(conv_pw_9_bn)

	conv_pw_9_relu = ReLU(max_value=6)(conv_pw_9_bn)
	layers.append(conv_pw_9_relu)

	if split_boundary == 'conv_dw_10':
		return layers
	conv_dw_10 = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_pw_9_relu)
	layers.append(conv_dw_10)

	conv_dw_10_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_dw_10)
	layers.append(conv_dw_10_bn)

	conv_dw_10_relu = ReLU(max_value=6)(conv_dw_10_bn)
	layers.append(conv_dw_10_relu)

	conv_pw_10 = Conv2D(filters=512,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_dw_10_relu)
	layers.append(conv_pw_10)

	conv_pw_10_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_pw_10)
	layers.append(conv_pw_10_bn)

	conv_pw_10_relu = ReLU(max_value=6)(conv_pw_10_bn)
	layers.append(conv_pw_10_relu)

	if split_boundary == 'conv_dw_11':
		return layers
	conv_dw_11 = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_pw_10_relu)
	layers.append(conv_dw_11)

	conv_dw_11_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_dw_11)
	layers.append(conv_dw_11_bn)

	conv_dw_11_relu = ReLU(max_value=6)(conv_dw_11_bn)
	layers.append(conv_dw_11_relu)

	conv_pw_11 = Conv2D(filters=512,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_dw_11_relu)
	layers.append(conv_pw_11)

	conv_pw_11_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_pw_11)
	layers.append(conv_pw_11_bn)

	conv_pw_11_relu = ReLU(max_value=6)(conv_pw_11_bn)
	layers.append(conv_pw_11_relu)

	conv_pad_12 = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(conv_pw_11_relu)
	layers.append(conv_pad_12)
	
	if split_boundary == 'conv_dw_12':
		return layers
	conv_dw_12 = DepthwiseConv2D(kernel_size=(3,3),
		padding='valid',
		data_format='channels_last',
		activation='linear',
		strides = (2, 2),
		use_bias=False)(conv_pad_12)
	layers.append(conv_dw_12)

	conv_dw_12_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_dw_12)
	layers.append(conv_dw_12_bn)

	conv_dw_12_relu = ReLU(max_value=6)(conv_dw_12_bn)
	layers.append(conv_dw_12_relu)

	conv_pw_12 = Conv2D(filters=1024,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_dw_12_relu)
	layers.append(conv_pw_12)

	conv_pw_12_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_pw_12)
	layers.append(conv_pw_12_bn)

	conv_pw_12_relu = ReLU(max_value=6)(conv_pw_12_bn)
	layers.append(conv_pw_12_relu)

	if split_boundary == 'conv_dw_13':
		return layers
	conv_dw_13 = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_pw_12_relu)
	layers.append(conv_dw_13)

	conv_dw_13_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_dw_13)
	layers.append(conv_dw_13_bn)

	conv_dw_13_relu = ReLU(max_value=6)(conv_dw_13_bn)
	layers.append(conv_dw_13_relu)

	conv_pw_13 = Conv2D(filters=1024,
		kernel_size=(1,1),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv_dw_13_relu)
	layers.append(conv_pw_13)

	conv_pw_13_bn = BatchNormalization(axis=3,
		momentum=0.99)(conv_pw_13)
	layers.append(conv_pw_13_bn)

	conv_pw_13_relu = ReLU(max_value=6)(conv_pw_13_bn)
	layers.append(conv_pw_13_relu)

	global_average_pooling2d = GlobalAveragePooling2D(data_format='channels_last')(conv_pw_13_relu)
	layers.append(global_average_pooling2d)

	# pred_1 = Dense(units=1000, activation='softmax')(global_average_pooling2d)
	# entire_model = tf.keras.Model(inputs=input_1, outputs=pred_1)
	return layers

def define_MobileNet_for_B(split_boundary, in_tensor):
	"""
	Define MobileNet model architecture
	"""
	layers = []
	split_boundary_candidates = ['conv_dw_1', 'conv_dw_2', 'conv_dw_3', 'conv_dw_4',
		'conv_dw_5', 'conv_dw_6', 'conv_dw_7', 'conv_dw_8', 'conv_dw_9', 'conv_dw_10',
		'conv_dw_11', 'conv_dw_12', 'conv_dw_13']
	cond = split_boundary_candidates.index(split_boundary)

	input_2 = tf.keras.Input(tensor=in_tensor)

	if cond == 0:
		conv1_relu = input_2
		layers.append(conv1_relu)

		conv_dw_1 = DepthwiseConv2D(kernel_size=(3,3),
		padding='same',
		data_format='channels_last',
		activation='linear',
		use_bias=False)(conv1_relu)
		layers.append(conv_dw_1)

		conv_dw_1_bn= BatchNormalization(axis=3,
			momentum=0.99)(conv_dw_1)
		layers.append(conv_dw_1_bn)
	
		conv_dw_1_relu = ReLU(max_value=6)(conv_dw_1_bn)
		layers.append(conv_dw_1_relu)	
	
		conv_pw_1 = Conv2D(filters=64,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_dw_1_relu)
		layers.append(conv_pw_1)
	
		conv_pw_1_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_pw_1)
		layers.append(conv_pw_1_bn)
	
		conv_pw_1_relu = ReLU(max_value=6)(conv_pw_1_bn)
		layers.append(conv_pw_1_relu)
	
		conv_pad_2 = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(conv_pw_1_relu)
		layers.append(conv_pad_2)
	if cond == 1:
		conv_pad_2 = input_2
		layers.append(conv_pad_2)
	if cond in range(2):
		conv_dw_2 = DepthwiseConv2D(kernel_size=(3,3),
			padding='valid',
			data_format='channels_last',
			activation='linear',
			strides = (2, 2),
			use_bias=False)(conv_pad_2)
		layers.append(conv_dw_2)
	
		conv_dw_2_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_dw_2)
		layers.append(conv_dw_2_bn)
	
		conv_dw_2_relu = ReLU(max_value=6)(conv_dw_2_bn)
		layers.append(conv_dw_2_relu)
	
		conv_pw_2 = Conv2D(filters=128,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_dw_2_relu)
		layers.append(conv_pw_2)

		conv_pw_2_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_pw_2)
		layers.append(conv_pw_2_bn)

		conv_pw_2_relu = ReLU(max_value=6)(conv_pw_2_bn)
		layers.append(conv_pw_2_relu)
	if cond == 2:
		conv_pw_2_relu = input_2
		layers.append(conv_pw_2_relu)
	if cond in range(3):
		conv_dw_3 = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_pw_2_relu)
		layers.append(conv_dw_3)

		conv_dw_3_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_dw_3)
		layers.append(conv_dw_3_bn)

		conv_dw_3_relu = ReLU(max_value=6)(conv_dw_3_bn)
		layers.append(conv_dw_3_relu)

		conv_pw_3 = Conv2D(filters=128,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_dw_3_relu)
		layers.append(conv_pw_3)

		conv_pw_3_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_pw_3)
		layers.append(conv_pw_3_bn)

		conv_pw_3_relu = ReLU(max_value=6)(conv_pw_3_bn)
		layers.append(conv_pw_3_relu)

		conv_pad_4 = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(conv_pw_3_relu)
		layers.append(conv_pad_4)
	if cond == 3:
		conv_pad_4 = input_2
		layers.append(conv_pad_4)
	if cond in range(4):
		conv_dw_4 = DepthwiseConv2D(kernel_size=(3,3),
			padding='valid',
			data_format='channels_last',
			activation='linear',
			strides = (2, 2),
			use_bias=False)(conv_pad_4)
		layers.append(conv_dw_4)

		conv_dw_4_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_dw_4)
		layers.append(conv_dw_4_bn)

		conv_dw_4_relu = ReLU(max_value=6)(conv_dw_4_bn)
		layers.append(conv_dw_4_relu)

		conv_pw_4 = Conv2D(filters=256,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_dw_4_relu)
		layers.append(conv_pw_4)

		conv_pw_4_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_pw_4)
		layers.append(conv_pw_4_bn)

		conv_pw_4_relu = ReLU(max_value=6)(conv_pw_4_bn)
		layers.append(conv_pw_4_relu)

	if cond == 4:
		conv_pw_4_relu = input_2
		layers.append(conv_pw_4_relu)
	if cond in range(5):
		conv_dw_5 = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_pw_4_relu)
		layers.append(conv_dw_5)
	
		conv_dw_5_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_dw_5)
		layers.append(conv_dw_5_bn)

		conv_dw_5_relu = ReLU(max_value=6)(conv_dw_5_bn)
		layers.append(conv_dw_5_relu)

		conv_pw_5 = Conv2D(filters=256,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_dw_5_relu)
		layers.append(conv_pw_5)

		conv_pw_5_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_pw_5)
		layers.append(conv_pw_5_bn)

		conv_pw_5_relu = ReLU(max_value=6)(conv_pw_5_bn)
		layers.append(conv_pw_5_relu)

		conv_pad_6 = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(conv_pw_5_relu)
		layers.append(conv_pad_6)

	if cond == 5:
		conv_pad_6 = input_2
		layers.append(conv_pad_6)
	if cond in range(6):
		conv_dw_6 = DepthwiseConv2D(kernel_size=(3,3),
			padding='valid',
			data_format='channels_last',
			activation='linear',
			strides = (2, 2),
			use_bias=False)(conv_pad_6)
		layers.append(conv_dw_6)

		conv_dw_6_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_dw_6)
		layers.append(conv_dw_6_bn)

		conv_dw_6_relu = ReLU(max_value=6)(conv_dw_6_bn)
		layers.append(conv_dw_6_relu)

		conv_pw_6 = Conv2D(filters=512,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_dw_6_relu)
		layers.append(conv_pw_6)

		conv_pw_6_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_pw_6)
		layers.append(conv_pw_6_bn)

		conv_pw_6_relu = ReLU(max_value=6)(conv_pw_6_bn)
		layers.append(conv_pw_6_relu)
	if cond == 6:
		conv_pw_6_relu = input_2
		layers.append(conv_pw_6_relu)		
	if cond in range(7):
		conv_dw_7 = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_pw_6_relu)
		layers.append(conv_dw_7)
	
		conv_dw_7_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_dw_7)
		layers.append(conv_dw_7_bn)

		conv_dw_7_relu = ReLU(max_value=6)(conv_dw_7_bn)
		layers.append(conv_dw_7_relu)

		conv_pw_7 = Conv2D(filters=512,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_dw_7_relu)
		layers.append(conv_pw_7)

		conv_pw_7_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_pw_7)
		layers.append(conv_pw_7_bn)

		conv_pw_7_relu = ReLU(max_value=6)(conv_pw_7_bn)
		layers.append(conv_pw_7_relu)

	if cond == 7:
		conv_pw_7_relu = input_2
		layers.append(conv_pw_7_relu)	
	if cond in range(8):
		conv_dw_8 = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_pw_7_relu)
		layers.append(conv_dw_8)
	
		conv_dw_8_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_dw_8)
		layers.append(conv_dw_8_bn)

		conv_dw_8_relu = ReLU(max_value=6)(conv_dw_8_bn)
		layers.append(conv_dw_8_relu)

		conv_pw_8 = Conv2D(filters=512,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_dw_8_relu)
		layers.append(conv_pw_8)

		conv_pw_8_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_pw_8)
		layers.append(conv_pw_8_bn)

		conv_pw_8_relu = ReLU(max_value=6)(conv_pw_8_bn)
		layers.append(conv_pw_8_relu)
	if cond == 8:
		conv_pw_8_relu = input_2
		layers.append(conv_pw_8_relu)	
	if cond in range(9):
		conv_dw_9 = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_pw_8_relu)
		layers.append(conv_dw_9)

		conv_dw_9_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_dw_9)
		layers.append(conv_dw_9_bn)

		conv_dw_9_relu = ReLU(max_value=6)(conv_dw_9_bn)
		layers.append(conv_dw_9_relu)

		conv_pw_9 = Conv2D(filters=512,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_dw_9_relu)
		layers.append(conv_pw_9)

		conv_pw_9_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_pw_9)
		layers.append(conv_pw_9_bn)

		conv_pw_9_relu = ReLU(max_value=6)(conv_pw_9_bn)
		layers.append(conv_pw_9_relu)
	if cond == 9:
		conv_pw_9_relu = input_2
		layers.append(conv_pw_9_relu)
	if cond in range(10):
		conv_dw_10 = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_pw_9_relu)
		layers.append(conv_dw_10)
	
		conv_dw_10_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_dw_10)
		layers.append(conv_dw_10_bn)

		conv_dw_10_relu = ReLU(max_value=6)(conv_dw_10_bn)
		layers.append(conv_dw_10_relu)

		conv_pw_10 = Conv2D(filters=512,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_dw_10_relu)
		layers.append(conv_pw_10)

		conv_pw_10_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_pw_10)
		layers.append(conv_pw_10_bn)

		conv_pw_10_relu = ReLU(max_value=6)(conv_pw_10_bn)
		layers.append(conv_pw_10_relu)

	if cond == 10:
		conv_pw_10_relu = input_2
		layers.append(conv_pw_10_relu)	
	if cond in range(11):
		conv_dw_11 = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_pw_10_relu)
		layers.append(conv_dw_11)
	
		conv_dw_11_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_dw_11)
		layers.append(conv_dw_11_bn)

		conv_dw_11_relu = ReLU(max_value=6)(conv_dw_11_bn)
		layers.append(conv_dw_11_relu)

		conv_pw_11 = Conv2D(filters=512,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_dw_11_relu)
		layers.append(conv_pw_11)

		conv_pw_11_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_pw_11)
		layers.append(conv_pw_11_bn)

		conv_pw_11_relu = ReLU(max_value=6)(conv_pw_11_bn)
		layers.append(conv_pw_11_relu)

		conv_pad_12 = ZeroPadding2D(padding=((0,1),(0,1)), data_format='channels_last')(conv_pw_11_relu)
		layers.append(conv_pad_12)
	if cond == 11:
		conv_pad_12 = input_2
		layers.append(conv_pad_12)		
	if cond in range(12):
		conv_dw_12 = DepthwiseConv2D(kernel_size=(3,3),
			padding='valid',
			data_format='channels_last',
			activation='linear',
			strides = (2, 2),
			use_bias=False)(conv_pad_12)
		layers.append(conv_dw_12)
	
		conv_dw_12_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_dw_12)
		layers.append(conv_dw_12_bn)

		conv_dw_12_relu = ReLU(max_value=6)(conv_dw_12_bn)
		layers.append(conv_dw_12_relu)

		conv_pw_12 = Conv2D(filters=1024,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_dw_12_relu)
		layers.append(conv_pw_12)

		conv_pw_12_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_pw_12)
		layers.append(conv_pw_12_bn)

		conv_pw_12_relu = ReLU(max_value=6)(conv_pw_12_bn)
		layers.append(conv_pw_12_relu)
	if cond == 12:
		conv_pw_12_relu = input_2
		layers.append(conv_pw_12_relu)		
	if cond in range(13):
		conv_dw_13 = DepthwiseConv2D(kernel_size=(3,3),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_pw_12_relu)
		layers.append(conv_dw_13)
	
		conv_dw_13_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_dw_13)
		layers.append(conv_dw_13_bn)

		conv_dw_13_relu = ReLU(max_value=6)(conv_dw_13_bn)
		layers.append(conv_dw_13_relu)

		conv_pw_13 = Conv2D(filters=1024,
			kernel_size=(1,1),
			padding='same',
			data_format='channels_last',
			activation='linear',
			use_bias=False)(conv_dw_13_relu)
		layers.append(conv_pw_13)

		conv_pw_13_bn = BatchNormalization(axis=3,
			momentum=0.99)(conv_pw_13)
		layers.append(conv_pw_13_bn)

		conv_pw_13_relu = ReLU(max_value=6)(conv_pw_13_bn)
		layers.append(conv_pw_13_relu)

		global_average_pooling2d = GlobalAveragePooling2D(data_format='channels_last')(conv_pw_13_relu)
		layers.append(global_average_pooling2d)

		# pred_1 = Dense(units=1000, activation='softmax')(global_average_pooling2d)
		# entire_model = tf.keras.Model(inputs=input_1, outputs=pred_1)

	return layers	

def split(full_model, split_boundary):
	layer_names = [layer.name for layer in full_model.layers]
	split_boundary_idx = layer_names.index(split_boundary) 

	layers = define_MobileNet_for_A(split_boundary, full_model.input)
	print(layers)
	full_A_model = tf.keras.Model(inputs=layers[0], outputs=layers[-1])
	print(len(full_A_model.layers))

	for idx in range(1, len(full_A_model.layers)):
		full_A_model.layers[idx].set_weights(full_model.layers[idx].get_weights())
	full_A_model.summary()

	layers = define_MobileNet_for_B(split_boundary, full_model.layers[split_boundary_idx].input)
	full_B_model = tf.keras.Model(inputs=layers[0], outputs=layers[-1])
	full_B_model.summary()
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

	# Pretrained MobileNet를 불러오고 그 개요를 출력
	# MobileNet의 classifier 부분은 불러오지 않는다. (i.e., include_top=False)
	full_model = tf.keras.applications.MobileNet(input_shape=(224,224,3), include_top=False, pooling='avg')
	print('Successfully imported full model')
	full_model.summary()

	full_A_model, full_B_model, layer_names, split_boundary_idx = split(full_model, args.split_boundary)
	print('Successfully split full model into part A and B')

	# full_A_model과 full_B_model을 각각 .h5 file으로 저장
	MobileNet_with_imagenet_full_A_path = os.path.join(args.output_dir, 'MobileNet_with_ImageNet_full_A.h5') 
	MobileNet_with_imagenet_full_A_pb_path = os.path.join(args.output_dir, 'MobileNet_with_ImageNet_full_A_pb') 
	MobileNet_with_imagenet_full_B_path = os.path.join(args.output_dir, 'MobileNet_with_ImageNet_full_B.h5') 
	
	full_A_model.save(MobileNet_with_imagenet_full_A_pb_path, save_format='tf')
	imported = tf.saved_model.load(MobileNet_with_imagenet_full_A_pb_path)

	tf.keras.models.save_model(full_A_model, MobileNet_with_imagenet_full_A_path)
	tf.keras.models.save_model(full_B_model, MobileNet_with_imagenet_full_B_path)
	print('Successfully saved part A to {}'.format(MobileNet_with_imagenet_full_A_path))
	print('Successfully saved part B to {}'.format(MobileNet_with_imagenet_full_B_path))

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
	MobileNet_quant_A_path = os.path.join(args.output_dir, 'MobileNet_with_ImageNet_quant_A.tflite')
	MobileNet_quant_B_path = os.path.join(args.output_dir, 'MobileNet_with_ImageNet_quant_B.tflite')
	open(MobileNet_quant_A_path, 'wb').write(quant_A_model)	
	print('Successfully saved the quantized part A to {}'.format(MobileNet_quant_A_path))
	open(MobileNet_quant_B_path, 'wb').write(quant_B_model)	
	print('Successfully saved the quantized part B to {}'.format(MobileNet_quant_B_path))

	# 사전에 edgetpu-compiler를 install하여야 함
	os.system('edgetpu_compiler {0}'.format(MobileNet_quant_A_path))
	os.system('edgetpu_compiler {0}'.format(MobileNet_quant_B_path))
	print('Successfully compiled the quantized parts for inference with edge TPU')

if __name__ == '__main__':
  main()
