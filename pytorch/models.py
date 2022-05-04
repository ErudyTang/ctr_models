import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config_parser import Config
from data_helper import TextDataset

def _init_input_modules(features, _dimension):
	module_dict = nn.ModuleDict()
	dimension_dict = {}
	for fea_conf in features:
		trans = fea_conf.trans
		if trans == None:
			continue
		params = fea_conf.params
		bucket_size, dimension = params
		if _dimension:
			dimension = _dimension
		m = nn.Embedding(bucket_size+1, dimension, padding_idx=-1)
		nn.init.xavier_uniform_(m.weight)
		module_dict[fea_conf.name] = m
		dimension_dict[fea_conf.name] = dimension

	return module_dict, dimension_dict

class LinearProjector(nn.Module):
	def __init__(self, features, _dimension=None):
		super().__init__()
		self.features = features
		self.module_dict, self.dimension_dict = _init_input_modules(features, _dimension)

	def forward(self, data):
		projections = []
		dimensions = []
		for fea_conf in self.features:
			trans = fea_conf.trans
			if trans == None:
				continue
			projections.append(self.module_dict[fea_conf.name](data[fea_conf.name]))
			dimensions.append(self.dimension_dict[fea_conf.name])
		return projections

class DNN(nn.Module):
	def __init__(self, features, hidden_units):
		super(DNN, self).__init__()
		self.proj = LinearProjector(features)
		in_dim = sum(list(self.proj.dimension_dict.values()))
		layers = []
		for out_dim in hidden_units:
			layers.append(nn.Linear(in_dim, out_dim))
			layers.append(nn.Dropout(p=0.1))
			layers.append(nn.ReLU())
			in_dim = out_dim
		layers.append(nn.Linear(in_dim, 1))
		layers.append(nn.Sigmoid())
		self.seq = nn.Sequential(*layers)

	def forward(self, data):
		x = torch.cat(self.proj(data), dim=1)
		return torch.squeeze(self.seq(x))

class CrossLayer(nn.Module):
	def __init__(self, units, n_layers):
		super(CrossLayer, self).__init__()
		self.n_layers = n_layers
		self.w_list = nn.ParameterList()
		self.b_list = nn.ParameterList()
		for _ in range(n_layers):
			self.w_list.append(nn.Parameter(torch.nn.init.normal_(torch.empty(units, 1))))
			self.b_list.append(nn.Parameter(torch.nn.init.normal_(torch.empty(units))))
	def forward(self, data):
		out = data
		for i in range(self.n_layers):
			out = data * torch.matmul(data, self.w_list[i]) + self.b_list[i] + data
		return out

class DeepCross(nn.Module):
	def __init__(self, features, hidden_units, n_cross):
		super(DeepCross, self).__init__()
		self.proj = LinearProjector(features)
		in_dim = sum(list(self.proj.dimension_dict.values()))
		self.cross_layer = CrossLayer(in_dim, n_cross)
		self.output_layer = nn.Linear(in_dim + hidden_units[-1], 1)
		dnn_layers = []
		for out_dim in hidden_units:
			dnn_layers.append(nn.Linear(in_dim, out_dim))
			dnn_layers.append(nn.Dropout(p=0.1))
			dnn_layers.append(nn.ReLU())
			in_dim = out_dim
		self.dnn_seq = nn.Sequential(*dnn_layers)

	def forward(self, data):
		x = torch.cat(self.proj(data), dim=1)
		cross = self.cross_layer(x)
		dnn = self.dnn_seq(x)
		return torch.squeeze(torch.sigmoid(self.output_layer(torch.cat([cross, dnn], dim=1))))

class FM(nn.Module):
	def __init__(self, features):
		super(FM, self).__init__()
		self.emb_proj = LinearProjector(features, 8)
		self.linear_proj = LinearProjector(features, 1)
		self.register_parameter(name='b', param=torch.nn.Parameter(torch.tensor(0.0)))

	def forward(self, data):
		linear_out = torch.cat(self.linear_proj(data), dim=1)
		embs = self.emb_proj(data)
		embs_square = []
		for emb in embs:
			embs_square.append(torch.mul(emb, emb))
		sum_square = torch.sum(torch.stack(embs, dim=1), dim=1)
		sum_square = torch.mul(sum_square, sum_square)
		square_sum = torch.sum(torch.stack(embs_square, dim=1), dim=1)
		emb_out = sum_square - square_sum
		return torch.sigmoid(torch.sum(linear_out, dim=1) + 0.5 * torch.sum(emb_out, dim=1) + self.b)

class DeepFM(nn.Module):
	def __init__(self, features, hidden_units):
		super(DeepFM, self).__init__()

		self.emb_proj = LinearProjector(features, 8)
		self.linear_proj = LinearProjector(features, 1)
		self.register_parameter(name='b', param=torch.nn.Parameter(torch.tensor(0.0)))

		in_dim = sum(list(self.emb_proj.dimension_dict.values()))
		layers = []
		for out_dim in hidden_units:
			layers.append(nn.Linear(in_dim, out_dim))
			layers.append(nn.Dropout(p=0.1))
			layers.append(nn.ReLU())
			in_dim = out_dim
		layers.append(nn.Linear(in_dim, 1))
		self.seq = nn.Sequential(*layers)

	def forward(self, data):
		linear_out = torch.cat(self.linear_proj(data), dim=1)
		embs = self.emb_proj(data)
		embs_square = []
		for emb in embs:
			embs_square.append(torch.mul(emb, emb))
		sum_square = torch.sum(torch.stack(embs, dim=1), dim=1)
		sum_square = torch.mul(sum_square, sum_square)
		square_sum = torch.sum(torch.stack(embs_square, dim=1), dim=1)
		emb_out = sum_square - square_sum
		fm = torch.sum(linear_out, dim=1) + 0.5 * torch.sum(emb_out, dim=1) + self.b

		dnn = torch.squeeze(self.seq(torch.cat(embs, dim=1)))

		return torch.sigmoid(fm + dnn)

class LR(nn.Module):
	def __init__(self, features):
		super(LR, self).__init__()
		self.proj = LinearProjector(features, 1)
		self.register_parameter(name='b', param=torch.nn.Parameter(torch.tensor(0.0)))

	def forward(self, data):
		x = torch.cat(self.proj(data), dim=1)
		return torch.sigmoid(torch.sum(x, dim=1) + self.b)

if __name__ == '__main__':

	config = Config('../conf/th_conf.json')
	dataset = TextDataset(['../data/val_th/val_100k.txt'], config)
	dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
	samples, labels = next(iter(dataloader))
	#print(samples)
	#print(labels)
	#lp = LinearProjector(config.features)
	#x = lp(samples)
	#print(x)
	#print(len(x))
	#dnn = DNN(config.features, [128, 64])
	#x = dnn(samples)
	#print(x)
	#print(x.shape)
	#lr = LR(config.features)
	#x = lr(samples)
	#print(x)
	#print(x.shape)
	#fm = FM(config.features)
	#x = fm(samples)
	#print(x)
	#print(x.shape)
	#dfm = DeepFM(config.features, [128, 64])
	#x = dfm(samples)
	#print(x)
	#print(x.shape)
	dcn = DeepCross(config.features, [128, 64], 2)
	x = dcn(samples)
	print(x)
	print(x.shape)

