#coding=utf-8
import os
import argparse
import logging
import tensorflow as tf
from data_helper import TextDataHelper
from estimator_config_parser import Config
from model_def.estimators import DNN, LR, WD, FM
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
	
	def _get_estimator(self):
		ckpt_dir = os.path.join(self.config.ckpt_dir, self.model_name)
		if self.model_name == 'dnn_custom':
			model = DNN.get_custom_estimator(ckpt_dir, self.config.embedding_columns, self.model_config.units)
		elif self.model_name == 'dnn':
			model = DNN.get_estimator(ckpt_dir, self.config.embedding_columns, self.model_config.units)
		elif self.model_name == 'lr':
			model = LR.get_estimator(ckpt_dir, self.config.categorical_columns)
		elif self.model_name == 'wd':
			model = WD.get_estimator(ckpt_dir, self.config.categorical_columns, self.config.embedding_columns, self.model_config.units)
		elif self.model_name == 'fm':
			model = FM.get_estimator(ckpt_dir, self.config.linear_columns, self.config.embedding_columns)
		return model

	def train(self, val=True):
		model = self._get_estimator()
		data_helper = TextDataHelper(self.config)
		train_file_list = get_files_indir(self.config.train_data_dir)
		for i in range(0, self.model_config.epoch):
			model.train(input_fn=lambda: data_helper.train_input_fn(train_file_list, batch_size=self.model_config.batch_size))
			if val:
				val_file_list = get_files_indir(self.config.val_data_dir)
				model.evaluate(input_fn=lambda: data_helper.train_input_fn(val_file_list, batch_size=self.model_config.batch_size))

	def export(self):
		pass

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model', choices=['dnn', 'dnn_custom', 'fm', 'lr', 'wd'], help='选择模型')
	parser.add_argument('config', help='模型训练的配置文件')
	args = parser.parse_args()

	trainer = Trainer(args.model, args.config)
	trainer.train()
