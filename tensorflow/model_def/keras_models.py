#coding=utf-8
import sys
import os
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.get_logger().setLevel(logging.ERROR)

class DNN(keras.Model):

	# pp_outputs：原始特征预处理后输出，预处理方式无限制
	def __init__(self, inputs, pp_outputs, hidden_units, name='DNN', **kwargs):
		super(DNN, self).__init__(name=name, **kwargs)

		self.input_layer = keras.Model(inputs, pp_outputs)
		self.dense_layers = [layers.Dense(u, activation='relu') for u in hidden_units]
		#self.bn_layers = [layers.BatchNormalization() for _ in hidden_units]
		self.dropout_layers = [layers.Dropout(.1) for _ in hidden_units]
		self.output_layer = layers.Dense(1, activation='sigmoid', name='prob')

	def call(self, inputs, training=False):
		x = self.input_layer(inputs)
		x = layers.Concatenate(axis=1)(list(x.values()))
		for i, layer in enumerate(self.dense_layers):
			x = layer(x)
			#x = self.bn_layers[i](x, training)
			x = self.dropout_layers[i](x, training)
		return self.output_layer(x)

class FM(keras.Model):

	# pp_linear_outputs：原始特征进行dimension=1 embedding
	# pp_embedding_outputs：原始特征进行dimension=n embedding
	def __init__(self, inputs, pp_linear_outputs, pp_embedding_outputs, name='FM', **kwargs):
		super(FM, self).__init__(name=name, **kwargs)

		self.linear_input_layer = keras.Model(inputs, pp_linear_outputs)
		self.embedding_input_layer = keras.Model(inputs, pp_embedding_outputs)
		self.b = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name='bias')
		#self.regularization = layers.ActivityRegularization(l2=0.5)

	def call(self, inputs):
		# linear part
		x = tf.add_n(list(self.linear_input_layer(inputs).values()))

		# embedding part
		embs = list(self.embedding_input_layer(inputs).values())
		embs_square = []
		for i, emb in enumerate(embs):
			#emb = self.regularization(emb); embs[i] = emb
			embs_square.append(tf.multiply(emb, emb))
		embs_sum = tf.add_n(embs)
		sum_square = tf.multiply(embs_sum, embs_sum)
		square_sum = tf.add_n(embs_square)
		y = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1, keepdims=True)
				
		return keras.activations.sigmoid(x + y + self.b)

class DeepFM(keras.Model):

	# pp_linear_outputs：原始特征进行dimension=1 embedding
	# pp_embedding_outputs：原始特征进行dimension=n embedding
	def __init__(self, inputs, pp_linear_outputs, pp_embedding_outputs, hidden_units, name='DeepFM', **kwargs):
		super(DeepFM, self).__init__(name=name, **kwargs)

		self.b = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name='bias')
		self.linear_input_layer = keras.Model(inputs, pp_linear_outputs)
		self.embedding_input_layer = keras.Model(inputs, pp_embedding_outputs)
		#self.regularization = layers.ActivityRegularization(l2=.5)
		self.dense_layers = [layers.Dense(u, activation='relu') for u in hidden_units]
		self.dropout_layers = [layers.Dropout(.1) for _ in hidden_units]
		self.dnn_output_layer = layers.Dense(1, activation=None, name='dnn_output')

	def call(self, inputs, training=False):
		# FM part
		x = tf.add_n(list(self.linear_input_layer(inputs).values()))
		embs = list(self.embedding_input_layer(inputs).values())
		embs_square = []
		for i, emb in enumerate(embs):
			#emb = self.regularization(emb); embs[i] = emb
			embs_square.append(tf.multiply(emb, emb))
		embs_sum = tf.add_n(embs)
		sum_square = tf.multiply(embs_sum, embs_sum)
		square_sum = tf.add_n(embs_square)
		fm = x + 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1, keepdims=True) + self.b

		# DNN part
		z = layers.Concatenate(axis=1)(embs)
		for i, layer in enumerate(self.dense_layers):
			z = layer(z)
			z = self.dropout_layers[i](z, training)
		dnn = self.dnn_output_layer(z)

		return keras.activations.sigmoid(fm + dnn)

class CrossLayer(keras.layers.Layer):
	def __init__(self, n_layers):
		super(CrossLayer, self).__init__()
		self.n_layers = n_layers
	def build(self, input_shape):
		self.w_list = []
		self.b_list = []
		for _ in range(self.n_layers):
			self.w_list.append(
				self.add_weight(
					shape=(input_shape[-1], 1),
					initializer="random_normal",
					trainable=True)
			)
			self.b_list.append(
				self.add_weight(
					shape=(input_shape[-1],), 
					initializer="random_normal", 
					trainable=True)
			)
	def call(self, inputs):
		outputs = inputs
		for i in range(self.n_layers):
			outputs = inputs * tf.matmul(outputs, self.w_list[i]) + self.b_list[i] + inputs
		return outputs

class DeepCross(keras.Model):

	# pp_outputs：原始特征预处理后输出，预处理方式无限制
	def __init__(self, inputs, pp_outputs, hidden_units, n_cross_layers, name='DeepCross', **kwargs):
		super(DeepCross, self).__init__(name=name, **kwargs)
		self.input_layer = keras.Model(inputs, pp_outputs)
		self.cross_layer = CrossLayer(n_cross_layers)
		#self.regularization = layers.ActivityRegularization(l2=.5)
		self.dense_layers = [layers.Dense(u, activation="relu") for u in hidden_units]
		self.dropout_layers = [layers.Dropout(.1) for _ in hidden_units]
		self.output_layer = layers.Dense(1, activation='sigmoid', name='prob')

	def call(self, inputs, training=False):
		x = self.input_layer(inputs)
		x = layers.Concatenate(axis=1)(list(x.values()))
		# Cross part
		cross = self.cross_layer(x)

		# DNN part
		dnn = x
		for i, layer in enumerate(self.dense_layers):
			dnn = layer(dnn)
			dnn = self.dropout_layers[i](dnn, training)

		return self.output_layer(tf.concat([cross, dnn], axis=1))

class LR(keras.Model):

	# pp_outputs：原始特征进行dimension=1 embedding
	def __init__(self, inputs, pp_outputs, name='LR', **kwargs):
		super(LR, self).__init__(name=name, **kwargs)

		self.input_layer = keras.Model(inputs, pp_outputs)
		self.b = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name='bias')

	def call(self, inputs):
		x = tf.add_n(list(self.input_layer(inputs).values()))
		return keras.activations.sigmoid(x + self.b)

class LR2(keras.Model):

	# pp_outputs：原始特征进行one hot预处理
	def __init__(self, inputs, pp_outputs, name='LR2', **kwargs):
		super(LR2, self).__init__(name=name, **kwargs)

		self.input_layer = keras.Model(inputs, pp_outputs)
		self.dense_layer = layers.Dense(1, activation='sigmoid')

	def call(self, inputs):
		x = self.input_layer(inputs)
		x = layers.Concatenate(axis=1)(list(x.values()))
		return self.dense_layer(x)
