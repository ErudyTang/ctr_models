import sys
import logging
import tensorflow as tf
from tensorflow import feature_column
from tensorflow import keras
from tensorflow.keras import layers
import json
import os
tf.get_logger().setLevel(logging.ERROR)

type_convert_map = {
	'string': str,
	'string_list': str,
	'int': int,
	'float': float
}

tf_type_map = {
	'string': tf.string,
	'string_list': tf.string,
	'int': tf.int64,
	'float': tf.float32
}

class ModelConfig(object):
	def __init__(self, m):
		self.name = str(m['name'])
		self.opt = m['optimizer']
		self.lr = m['learning_rate']
		if 'hidden_units' in m:
			self.units = [int(v) for v in m['hidden_units'].split(',')]
		if 'n_cross_layers' in m:
			self.n_cross = m['n_cross_layers']
		self.batch_size = m['batch_size']
		self.epoch = m['epoch']

class FeatureConfig(object):
	def __init__(self, m):
		self.name = str(m['name'])
		self.type = m['type']
		self.default = type_convert_map[self.type](m['default'])
		self.trans = None
		if 'trans' in m: self.trans = m['trans']
		if 'params' in m: self.params = m['params']

class Config(object):
	def __init__(self, config_file):
		self._parse_config(config_file)
		self._get_feature_columns()
		self._get_preprocessing_inputs_outputs()

	def _parse_config(self, config_file):
		with open(config_file) as f:
			m = json.load(f)
		home_dir = m['home_dir']
		self.ckpt_dir = os.path.join(home_dir, m['ckpt_dir'])
		self.model_dir = os.path.join(home_dir, m['model_dir'])
		self.log_dir = os.path.join(home_dir, m['log_dir'])
		self.train_data_dir = os.path.join(home_dir, m['train_data_dir'])
		self.val_data_dir = os.path.join(home_dir, m['val_data_dir'])
		self.models = {}
		self.label = None
		self.features = []
		self.need_split_features = []
		for mod in m['models']:
			mod_obj = ModelConfig(mod)
			self.models[mod_obj.name] = mod_obj
		for fea in m['features']:
			if 'label' in fea and fea['label']:
				self.label = FeatureConfig(fea)
				continue
			if 'ignore' in fea and fea['ignore']:
				continue
			fea_obj = FeatureConfig(fea)
			self.features.append(fea_obj)
			if fea_obj.type == 'string_list':
				self.need_split_features.append(fea_obj.name)

	def _get_feature_columns(self):
		self.embedding_columns = []
		self.linear_columns = []
		self.categorical_columns = []
		for fea_conf in self.features:
			trans = fea_conf.trans
			if trans == None:
				continue
			params = fea_conf.params
			bucket_size, dimension = params
			fea = feature_column.categorical_column_with_hash_bucket(
				fea_conf.name, hash_bucket_size=bucket_size, dtype=tf_type_map[fea_conf.type])
			self.categorical_columns.append(fea)
			emb_fea = feature_column.embedding_column(
				fea, dimension=dimension, combiner='mean')
			self.embedding_columns.append(emb_fea)
			emb_fea = feature_column.embedding_column(
				fea, dimension=1, combiner='mean')
			self.linear_columns.append(emb_fea)

	def _get_preprocessing_inputs_outputs(self):
		self.inputs = {}
		self.embedding_outputs = {}
		self.linear_outputs = {}
		self.one_hot_outputs = {}
		for fea_conf in self.features:
			trans = fea_conf.trans
			if trans == None:
				continue
			self.inputs[fea_conf.name] = tf.keras.Input(shape=(), dtype=tf_type_map[fea_conf.type])

			params = fea_conf.params
			bucket_size, dimension = params
			fea = layers.experimental.preprocessing.Hashing(bucket_size)(self.inputs[fea_conf.name])
			one_hot_fea = layers.experimental.preprocessing.CategoryEncoding(bucket_size, output_mode='binary', sparse=False)(fea)
			self.one_hot_outputs[fea_conf.name] = one_hot_fea
			emb_fea = layers.Embedding(bucket_size, dimension)(fea)
			self.embedding_outputs[fea_conf.name] = emb_fea
			emb_fea = layers.Embedding(bucket_size, 1)(fea)
			self.linear_outputs[fea_conf.name] = emb_fea
	
	def get_model_input_columns(self):
		input_column_map = {}
		for fea_conf in self.features:
			input_column_map[str(fea_conf.name)] = tf_type_map[fea_conf.type]
		return input_column_map
