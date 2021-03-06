from keras.layers import *
import tensorflow as tf
import numpy as np

NUM_STATE = 16
GRID_SIZE = 4
POWER = 12

table = {2 ** i: i for i in range(1, POWER)}
table[0] = 0

#NUM_STATE, no mask
def make_input_0(grid):
	return grid

#NUM_STATE * POWER , mask
def make_input_1(grid):
	g0 = grid
	r = np.zeros(shape=(POWER, NUM_STATE))
	for i in range(NUM_STATE):
		v = g0[i]
		r[table[v], i] = 1
	return r.flatten()

#GRID_SIZE * GRID_SIZE * 1, no_mask
def make_input_2(grid):
	return np.reshape(grid, (GRID_SIZE, GRID_SIZE, 1))

#GRID_SIZE * GRID_SIZE * POWER, mask
def make_input_3(grid):
	g0 = np.reshape(grid, (GRID_SIZE, GRID_SIZE))
	r = np.zeros(shape=(GRID_SIZE, GRID_SIZE, POWER))
	for i in range(GRID_SIZE):
		for j in range(GRID_SIZE):
			v = g0[i, j]
			r[i, j, table[v]] = 1
	return r


# input_layer, last_layer, placeholder, make_input, none_state
def only_dense_5_layers_256(mask):
	shape = NUM_STATE * POWER if mask else NUM_STATE

	input_layer = Input(batch_shape=(None, shape))
	l_dense0 = Dense(256, activation='relu')(input_layer)
	l_dense1 = Dense(128, activation='relu')(l_dense0)
	l_dense2 = Dense(64, activation='relu')(l_dense1)
	l_dense3 = Dense(32, activation='relu')(l_dense2)
	last_layer = Dense(16, activation='relu')(l_dense3)

	NONE_STATE = np.zeros(shape)
	s = tf.placeholder(tf.float32, shape=(None, shape))
	make_input = make_input_1 if mask else make_input_0

	return input_layer, last_layer, s, make_input, NONE_STATE

def conv2x2_layer_and_3_dense(mask):
	channels = POWER if mask else 1
	shape = (GRID_SIZE, GRID_SIZE, channels)

	input_layer = Input(shape=shape)
	conv_layer = Convolution2D(100, (2, 2), activation='relu')(input_layer)
	ft = Flatten()(conv_layer)
	l_dense = Dense(256, activation='relu')(ft)
	l_dense2 = Dense(64, activation='relu')(l_dense)
	last_layer = Dense(32, activation='relu')(l_dense2)

	NONE_STATE = np.zeros(shape=shape)
	make_input = make_input_3 if mask else make_input_2
	s = tf.placeholder(tf.float32, shape=(None, GRID_SIZE, GRID_SIZE, channels))

	return input_layer, last_layer, s, make_input, NONE_STATE

def two_conv_rect_layers_and_3_dense(mask):
	channels = POWER if mask else 1
	shape = (GRID_SIZE, GRID_SIZE, channels)

	input_layer = Input(shape=shape)
	conv_layer21 = Convolution2D(20, (2, 1), activation='relu')(input_layer)
	conv_layer12 = Convolution2D(20, (1, 2), activation='relu')(input_layer)
	ft = Flatten()
	merge_layer = concatenate([ft(conv_layer21), ft(conv_layer12)])
	l_dense = Dense(256, activation='relu')(merge_layer)
	l_dense2 = Dense(64, activation='relu')(l_dense)
	last_layer = Dense(32, activation='relu')(l_dense2)

	NONE_STATE = np.zeros(shape=shape)
	make_input = make_input_3 if mask else make_input_2
	s = tf.placeholder(tf.float32, shape=(None, GRID_SIZE, GRID_SIZE, channels))

	return input_layer, last_layer, s, make_input, NONE_STATE

def only_dense_6_layers_512(mask):
	shape = NUM_STATE * POWER if mask else NUM_STATE

	input_layer = Input(batch_shape=(None, shape))
	l_dense0 = Dense(512, activation='relu')(input_layer)
	l_dense1 = Dense(256, activation='relu')(l_dense0)
	l_dense2 = Dense(128, activation='relu')(l_dense1)
	l_dense3 = Dense(64, activation='relu')(l_dense2)
	l_dense4 = Dense(32, activation='relu')(l_dense3)
	last_layer = Dense(16, activation='relu')(l_dense4)

	NONE_STATE = np.zeros(shape)
	s = tf.placeholder(tf.float32, shape=(None, shape))
	make_input = make_input_1 if mask else make_input_0

	return input_layer, last_layer, s, make_input, NONE_STATE

def two_dense_256(mask):
	shape = NUM_STATE * POWER if mask else NUM_STATE

	input_layer = Input(batch_shape=(None, shape))
	l_dense0 = Dense(256, activation='relu')(input_layer)
	last_layer = Dense(256, activation='relu')(l_dense0)

	NONE_STATE = np.zeros(shape)
	s = tf.placeholder(tf.float32, shape=(None, shape))
	make_input = make_input_1 if mask else make_input_0

	return input_layer, last_layer, s, make_input, NONE_STATE

def two_conv_rect_layers_merge_with_input_and_3_dense(mask):
	channels = POWER if mask else 1
	shape = (GRID_SIZE, GRID_SIZE, channels)

	input_layer = Input(shape=shape)
	conv_layer21 = Convolution2D(20, (2, 1), activation='relu')(input_layer)
	conv_layer12 = Convolution2D(20, (1, 2), activation='relu')(input_layer)
	ft = Flatten()
	merge_layer = concatenate([ft(conv_layer21), ft(conv_layer12), ft(input_layer)])
	l_dense = Dense(256, activation='relu')(merge_layer)
	l_dense2 = Dense(64, activation='relu')(l_dense)
	last_layer = Dense(32, activation='relu')(l_dense2)

	NONE_STATE = np.zeros(shape=shape)
	make_input = make_input_3 if mask else make_input_2
	s = tf.placeholder(tf.float32, shape=(None, GRID_SIZE, GRID_SIZE, channels))

	return input_layer, last_layer, s, make_input, NONE_STATE

def conv2x2_layer_and_2_dense_512(mask):
	channels = POWER if mask else 1
	shape = (GRID_SIZE, GRID_SIZE, channels)

	input_layer = Input(shape=shape)
	conv_layer = Convolution2D(512, (2, 2), activation='relu')(input_layer)
	ft = Flatten()(conv_layer)
	l_dense = Dense(512, activation='relu')(ft)
	last_layer = Dense(512, activation='relu')(l_dense)

	NONE_STATE = np.zeros(shape=shape)
	make_input = make_input_3 if mask else make_input_2
	s = tf.placeholder(tf.float32, shape=(None, GRID_SIZE, GRID_SIZE, channels))

	return input_layer, last_layer, s, make_input, NONE_STATE

def two_dense_1024(mask):
	shape = NUM_STATE * POWER if mask else NUM_STATE

	input_layer = Input(batch_shape=(None, shape))
	l_dense0 = Dense(1024, activation='relu')(input_layer)
	last_layer = Dense(1024, activation='relu')(l_dense0)

	NONE_STATE = np.zeros(shape)
	s = tf.placeholder(tf.float32, shape=(None, shape))
	make_input = make_input_1 if mask else make_input_0

	return input_layer, last_layer, s, make_input, NONE_STATE


def four_conv_rect_512_layers(mask):
	channels = POWER if mask else 1
	shape = (GRID_SIZE, GRID_SIZE, channels)

	input_layer = Input(shape=shape)
	conv_layer_a = Convolution2D(512, (2, 1), activation='relu')(input_layer)
	conv_layer_b = Convolution2D(512, (1, 2), activation='relu')(input_layer)
	conv_layer_ab = Convolution2D(512, (1, 2), activation='relu')(conv_layer_a)
	conv_layer_ba = Convolution2D(512, (2, 1), activation='relu')(conv_layer_b)
	ft = Flatten()
	merge_layer = concatenate([ft(x) for x in[conv_layer_ab, conv_layer_ba, conv_layer_a, conv_layer_b]])
	last_layer = Dense(32, activation='relu')(merge_layer)

	NONE_STATE = np.zeros(shape=shape)
	make_input = make_input_3 if mask else make_input_2
	s = tf.placeholder(tf.float32, shape=(None, GRID_SIZE, GRID_SIZE, channels))

	return input_layer, last_layer, s, make_input, NONE_STATE

def conv2x2_only(mask):
	channels = POWER if mask else 1
	shape = (GRID_SIZE, GRID_SIZE, channels)

	input_layer = Input(shape=shape)
	conv_layer = Convolution2D(4096, (2, 2), activation='relu')(input_layer)
	last_layer = Flatten()(conv_layer)

	NONE_STATE = np.zeros(shape=shape)
	make_input = make_input_3 if mask else make_input_2
	s = tf.placeholder(tf.float32, shape=(None, GRID_SIZE, GRID_SIZE, channels))

	return input_layer, last_layer, s, make_input, NONE_STATE


def conv2x2_1024_and_two_dense_1024(mask):
	channels = POWER if mask else 1
	shape = (GRID_SIZE, GRID_SIZE, channels)

	input_layer = Input(shape=shape)
	conv_layer = Convolution2D(1024, (2, 2), activation='relu')(input_layer)
	ft = Flatten()(conv_layer)
	l_dense0 = Dense(1024, activation='relu')(ft)
	last_layer = Dense(1024, activation='relu')(l_dense0)

	NONE_STATE = np.zeros(shape=shape)
	make_input = make_input_3 if mask else make_input_2
	s = tf.placeholder(tf.float32, shape=(None, GRID_SIZE, GRID_SIZE, channels))

	return input_layer, last_layer, s, make_input, NONE_STATE

def five_dense_1024(mask):
	shape = NUM_STATE * POWER if mask else NUM_STATE

	input_layer = Input(batch_shape=(None, shape))
	l_dense0 = Dense(1024, activation='relu')(input_layer)
	l_dense1 = Dense(1024, activation='relu')(l_dense0)
	l_dense2 = Dense(1024, activation='relu')(l_dense1)
	l_dense3 = Dense(1024, activation='relu')(l_dense2)
	last_layer = Dense(1024, activation='relu')(l_dense3)

	NONE_STATE = np.zeros(shape)
	s = tf.placeholder(tf.float32, shape=(None, shape))
	make_input = make_input_1 if mask else make_input_0

	return input_layer, last_layer, s, make_input, NONE_STATE

def two_conv_rect_512_layers_merge_with_input_and_one_dense(mask):
	channels = POWER if mask else 1
	shape = (GRID_SIZE, GRID_SIZE, channels)

	input_layer = Input(shape=shape)
	conv_layer21 = Convolution2D(512, (2, 1), activation='relu')(input_layer)
	conv_layer12 = Convolution2D(512, (1, 2), activation='relu')(input_layer)
	ft = Flatten()
	merge_layer = concatenate([ft(conv_layer21), ft(conv_layer12), ft(input_layer)])
	last_layer = Dense(512, activation='relu')(merge_layer)

	NONE_STATE = np.zeros(shape=shape)
	make_input = make_input_3 if mask else make_input_2
	s = tf.placeholder(tf.float32, shape=(None, GRID_SIZE, GRID_SIZE, channels))

	return input_layer, last_layer, s, make_input, NONE_STATE


def two_conv_rect_1024_layers_merge_with_input_and_one_dense(mask):
	channels = POWER if mask else 1
	shape = (GRID_SIZE, GRID_SIZE, channels)

	input_layer = Input(shape=shape)
	conv_layer21 = Convolution2D(1024, (2, 1), activation='relu')(input_layer)
	conv_layer12 = Convolution2D(1024, (1, 2), activation='relu')(input_layer)
	ft = Flatten()
	merge_layer = concatenate([ft(conv_layer21), ft(conv_layer12), ft(input_layer)])
	last_layer = Dense(1024, activation='relu')(merge_layer)

	NONE_STATE = np.zeros(shape=shape)
	make_input = make_input_3 if mask else make_input_2
	s = tf.placeholder(tf.float32, shape=(None, GRID_SIZE, GRID_SIZE, channels))

	return input_layer, last_layer, s, make_input, NONE_STATE

def two_conv_128_layers_merge_with_input_and_one_dense(mask):
	channels = POWER if mask else 1
	shape = (GRID_SIZE, GRID_SIZE, channels)

	input_layer = Input(shape=shape)
	conv_layer1 = Convolution2D(128, (2, 2), activation='relu')(input_layer)
	conv_layer2 = Convolution2D(128, (3, 3), activation='relu')(conv_layer1)
	ft = Flatten()
	merge_layer = concatenate([ft(conv_layer1), ft(conv_layer2), ft(input_layer)])
	last_layer = Dense(54, activation='relu')(merge_layer)

	NONE_STATE = np.zeros(shape=shape)
	make_input = make_input_3 if mask else make_input_2
	s = tf.placeholder(tf.float32, shape=(None, GRID_SIZE, GRID_SIZE, channels))

	return input_layer, last_layer, s, make_input, NONE_STATE

NNs = [
	only_dense_5_layers_256,
	conv2x2_layer_and_3_dense,
	two_conv_rect_layers_and_3_dense,
	only_dense_6_layers_512,
	two_dense_256,
	two_conv_rect_layers_merge_with_input_and_3_dense,

	conv2x2_layer_and_2_dense_512,
	two_dense_1024,

	four_conv_rect_512_layers,
	conv2x2_only,
	conv2x2_1024_and_two_dense_1024,
	five_dense_1024,
	two_conv_rect_512_layers_merge_with_input_and_one_dense,
	two_conv_rect_1024_layers_merge_with_input_and_one_dense,
	two_conv_128_layers_merge_with_input_and_one_dense
]

def getNN(num=0, mask=False):
	return NNs[num](mask)
