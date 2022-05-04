#coding=utf-8
import os
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.get_logger().setLevel(logging.ERROR)

#基于feature column实现的keras model

class FCDNN(keras.Model):

	def __init__(self, feature_columns, hidden_units, name='FCDNN', **kwargs):
		super(FCDNN, self).__init__(name=name, **kwargs)

		self.input_layer = layers.DenseFeatures(feature_columns, name='input')
		self.dense_layers = [layers.Dense(u, activation='relu') for u in hidden_units]
		self.output_layer = layers.Dense(1, activation='sigmoid', name='prob')

	def call(self, inputs):
		x = self.input_layer(inputs)
		for layer in self.dense_layers:
			x = layer(x)
		x = self.output_layer(x)
		return x

class FCFM(keras.Model):

	def __init__(self, linear_columns, embedding_columns, name='FCFM', **kwargs):
		super(FCFM, self).__init__(name=name, **kwargs)

		self.linear_input_layer = layers.DenseFeatures(linear_columns)
		self.embedding_input_layers = [layers.DenseFeatures(column) for column in embedding_columns]
		self.b = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name='bias')
		#self.regularization = layers.ActivityRegularization(l2=0.5)

	def call(self, inputs):
		linear_output = self.linear_input_layer(inputs)
		#linear_output = self.regularization(linear_output)
		linear_output = tf.reduce_sum(linear_output, axis=1, keepdims=True)
		embeddings = []
		embeddings_square = []
		for layer in self.embedding_input_layers:
			embedding = layer(inputs)
			#embedding = self.regularization(embedding)
			embeddings.append(embedding)
			embeddings_square.append(tf.multiply(embedding, embedding))
		embeddings_sum = tf.add_n(embeddings)
		sum_square = tf.multiply(embeddings_sum, embeddings_sum)
		square_sum = tf.add_n(embeddings_square)
		embedding_output = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), axis=1, keepdims=True)
				
		x = keras.activations.sigmoid(linear_output + embedding_output + self.b)
		return x

class FCLR(keras.Model):

	def __init__(self, feature_columns, name='FCLR', **kwargs):
		super(FCLR, self).__init__(name=name, **kwargs)

		self.input_layer = layers.DenseFeatures(feature_columns)
		self.b = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name='bias')

	def call(self, inputs):
		linear_output = tf.reduce_sum(self.input_layer(inputs), axis=1, keepdims=True)
		x = keras.activations.sigmoid(linear_output + self.b)
		return x
