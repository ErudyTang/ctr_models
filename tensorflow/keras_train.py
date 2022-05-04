#coding=utf-8
import os
import argparse
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime, date, timedelta
from data_helper import TextDataHelper
from config_parser import Config
from model_def.fc_models import FCDNN, FCFM, FCLR
from model_def.keras_models import DNN, FM, DeepFM, DeepCross, LR, LR2
tf.get_logger().setLevel(logging.ERROR)

def get_files_indir(data_dir):
	file_list = []
	for filename in os.listdir(data_dir):
		file_list.append(os.path.join(data_dir, filename))
		file_list.sort()
	return file_list

class Trainer(object):

	def __init__(self, model_name, config):
		self.model_name = model_name
		self.config = Config(config)
		self.model_config = self.config.models[model_name]
	
	def _get_callbacks(self, checkpoint=True, tensorboard=True, early_stopping=False):
		callbacks = []
		if checkpoint:
			callbacks.append(
				keras.callbacks.ModelCheckpoint(
					filepath=os.path.join(self.config.ckpt_dir, self.model_name, '{epoch:02d}-{val_auc:.2f}'),
					save_best_only=True,  # Only save a model if `monitor` has improved.
					monitor='val_auc',
					mode='max',
					verbose=1,
				)
			)
		if tensorboard:
			callbacks.append(
				keras.callbacks.TensorBoard(
					log_dir=os.path.join(self.config.log_dir, self.model_name),
					histogram_freq=0,  # How often to log histogram visualizations
					embeddings_freq=0,  # How often to log embedding visualizations
					update_freq="epoch",
				)
			)
		if early_stopping:
			callbacks.append(
				keras.callbacks.EarlyStopping(
					monitor='val_auc',
					mode='max',
					patience=2,
					verbose=1,
				)
			)
		return callbacks

	def _get_optimizer(self):
		lr = self.model_config.lr
		opt = self.model_config.opt
		if opt == 'adam':
			return keras.optimizers.Adam(lr)
		elif opt == 'ftrl':
			return keras.optimizers.Ftrl(lr)
		else:
			return None

	def _get_compiled_model(self):
		if self.model_name == 'fm':
			#model = FCFM(self.config.linear_columns, self.config.embedding_columns)
			model = FM(self.config.inputs, self.config.linear_outputs, self.config.embedding_outputs)
		elif self.model_name == 'dfm':
			model = DeepFM(self.config.inputs, self.config.linear_outputs, self.config.embedding_outputs, self.model_config.units)
		elif self.model_name == 'dnn':
			#model = FCDNN(self.config.embedding_columns, self.model_config.units)
			model = DNN(self.config.inputs, self.config.embedding_outputs, self.model_config.units)
		elif self.model_name == 'lr':
			#model = FCLR(self.config.linear_columns)
			model = LR(self.config.inputs, self.config.linear_outputs)
			#model = LR2(self.config.inputs, self.config.one_hot_outputs)
		elif self.model_name == 'dcn':
			model = DeepCross(self.config.inputs, self.config.embedding_outputs, self.model_config.units, self.model_config.n_cross)
		model.compile(
			optimizer=self._get_optimizer(),
			loss=keras.losses.BinaryCrossentropy(),
			metrics=[
					keras.metrics.AUC(name='auc'),
				],
		)
		return model

	def _get_train_dataset(self, val):
		data_helper = TextDataHelper(self.config)
		train_file_list = get_files_indir(self.config.train_data_dir)
		train_dataset = data_helper.train_input_fn(train_file_list, batch_size=self.model_config.batch_size)
		val_dataset = None
		if val:
			val_file_list = get_files_indir(self.config.val_data_dir)
			val_dataset = data_helper.train_input_fn(val_file_list, batch_size=self.model_config.batch_size)
		return train_dataset, val_dataset

	def train(self, val=True):
		model = self._get_compiled_model()
		train_dataset, val_dataset = self._get_train_dataset(val)
		callbacks = self._get_callbacks(checkpoint=False, tensorboard=True, early_stopping=False)
		history = model.fit(
				train_dataset,
				validation_data=val_dataset,
				#validation_steps=10,
				epochs=self.model_config.epoch,
				callbacks=callbacks
			)
		model.summary()

	def export(self):
		model = self._get_compiled_model()
		model.save(os.path.join(self.config.model_dir, self.model_name))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model', choices=['dnn', 'fm', 'dfm', 'dcn', 'lr'], help='选择模型')
	parser.add_argument('config', help='模型训练的配置文件')
	args = parser.parse_args()

	trainer = Trainer(args.model, args.config)
	trainer.train()
