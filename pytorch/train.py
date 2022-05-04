import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from sklearn.metrics import roc_auc_score

from config_parser import Config
from data_helper import TextDataset
from models import DNN, DeepCross, FM, DeepFM, LR

def get_files_indir(data_dir):
	file_list = []
	for filename in os.listdir(data_dir):
		file_list.append(os.path.join(data_dir, filename))
		file_list.sort()
	return file_list

def train_loop(dataloader, model, loss_fn, optimizer):
	model.train()
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	train_loss = 0
	prob_all = []
	y_all = []

	for batch, (X, y) in enumerate(dataloader):
		# Compute prediction and loss
		pred = model(X)
		loss = loss_fn(pred, y.float())
		train_loss += loss.item()
		prob_all.extend(pred.tolist())
		y_all.extend(y.tolist())

		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	train_loss /= num_batches
	train_auc = roc_auc_score(y_all, prob_all)
	print(f"loss: {train_loss:>7.4f}  auc: {train_auc:>7.4f}\n---------------------------")
	return {'train_loss': train_loss, 'train_auc': train_auc}

def val_loop(dataloader, model, loss_fn):
	model.eval()
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	val_loss = 0
	prob_all = []
	y_all = []

	with torch.no_grad():
		for X, y in dataloader:
			pred = model(X)
			val_loss += loss_fn(pred, y.float()).item()
			prob_all.extend(pred.tolist())
			y_all.extend(y.tolist())

	val_loss /= num_batches
	val_auc = roc_auc_score(y_all, prob_all)
	print(f"loss: {val_loss:>7.4f}  auc: {val_auc:>7.4f}\n---------------------------")
	return {'val_loss': val_loss, 'val_auc': val_auc}

def train(model_name, model_config):
	config = Config(model_config)
	tb = SummaryWriter(comment='_'+model_name)
	model_config = config.models[model_name]

	train_dataset = TextDataset(get_files_indir(config.train_data_dir), config)
	train_dataloader = DataLoader(train_dataset, batch_size=model_config.batch_size, shuffle=True, num_workers=10)
	val_dataset = TextDataset(get_files_indir(config.val_data_dir), config)
	val_dataloader = DataLoader(val_dataset, batch_size=model_config.batch_size, shuffle=False, num_workers=10)

	if model_name == 'dnn':
		model = DNN(config.features, model_config.units)
	elif model_name == 'dcn':
		model = DeepCross(config.features, model_config.units, model_config.n_cross)
	elif model_name == 'fm':
		model = FM(config.features)
	elif model_name == 'dfm':
		model = DeepFM(config.features, model_config.units)
	elif model_name == 'lr':
		model = LR(config.features)

	loss_fn = nn.BCELoss()

	opt = model_config.opt
	lr = model_config.lr
	
	if opt == 'adam':
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	for e in range(model_config.epoch):
		print(f"Epoch {e+1} Train")
		metrics = train_loop(train_dataloader, model, loss_fn, optimizer)
		for k, v in metrics.items():
			tb.add_scalar(k, v, e)
		print(f"Epoch {e+1} Val")
		metrics = val_loop(val_dataloader, model, loss_fn)
		for k, v in metrics.items():
			tb.add_scalar(k, v, e)
	print("Done!")
	tb.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model', choices=['dnn', 'dcn', 'fm', 'dfm', 'lr'], help='选择模型')
	parser.add_argument('model_config', help='模型训练的配置文件')
	args = parser.parse_args()
	train(args.model, args.model_config)
