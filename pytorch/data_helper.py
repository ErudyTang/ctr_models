import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config_parser import Config

class TextDataset(Dataset):
	def __init__(self, file_names, config):
		super(TextDataset, self).__init__()
		self.config = config
		self.csv_column_keys = [self.config.label.name]
		for fea in self.config.features:
		    self.csv_column_keys.append(fea.name)

		data = pd.DataFrame(columns=self.csv_column_keys, dtype='Int64')
		for file_name in file_names:
			df = pd.read_csv(file_name,
					sep='\t',
					names=self.csv_column_keys,
					dtype='Int64')
			data = pd.concat([data, df], axis=0)
		self.labels = data.pop(self.config.label.name)
		self.samples = data

	def __len__(self):
		return self.labels.size

	def __getitem__(self, idx):
		label = torch.tensor(self.labels.iloc[idx])
		features = dict(zip(self.csv_column_keys[1:], torch.tensor(self.samples.iloc[idx].tolist())))
		return features, label
		
if __name__ == '__main__':

	config = Config('../../conf/th_model_conf.json')
	dataset = TextDataset(['../../data/train_th/train_1m.txt', '../../data/val_th/val_100k.txt'], config)
	print(len(dataset))
	dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
	samples, labels = next(iter(dataloader))
	print(samples)
	print(labels)
