import sys
import math
from collections import defaultdict
from sklearn.metrics import roc_auc_score

def sign(x):
	if x > 0:
		return 1
	return -1

def sigmoid(x):
	return 1.0 / (1.0 + math.exp(-x))

class LR(object):
	def __init__(self,
			train_file, val_file=None, opt='ftrl', alpha=0.05, beta=1.0, lambda1=0.1, lambda2=1.0,
			per_coordinate_lr=False, with_bias=True, log_size=100000
			):
		self.train_file = train_file
		self.val_file = val_file
		self.opt = opt
		self.w = defaultdict(float)
		self.z = defaultdict(float)
		self.n = defaultdict(float)
		self.a = alpha
		self.b = beta
		self.l1 = lambda1
		self.l2 = lambda2
		self.per_coordinate_lr = per_coordinate_lr
		self.with_bias = with_bias
		self.log_size = log_size
		self.val_label_list, self.val_feas_list = self._load_data(self.val_file)

	def _sgd_update(self, label, feas):
		pred = self._predict(feas)
		for k, v in feas.items():
			g = (pred - label) * v
			if self.per_coordinate_lr:
				self.n[k] += g**2
				self.w[k] -= self.a / (self.b + (self.n[k])**0.5) * g
			else:
				self.w[k] -= self.a * g
		return self._loss(label, pred)

	def _ftrl_update(self, label, feas):
		for k, v in feas.items():
			if v == 0.0:
				continue
			if abs(self.z[k]) <= self.l1:
				self.w[k] = 0.0
			else:
				self.w[k] = -(self.z[k] - sign(self.z[k]) * self.l1) / ((self.b + self.n[k]**0.5) / self.a + self.l2)
		pred = self._predict(feas)
		for k, v in feas.items():
			g = (pred - label) * v
			r = ((self.n[k] + g**2)**0.5 - (self.n[k])**0.5) / self.a
			self.z[k] += g - r * self.w[k]
			self.n[k] += g**2
		return self._loss(label, pred)

	def _loss(self, label, pred):
		return -label * math.log(pred) - (1 - label) * math.log(1 - pred)

	def _parse_input(self, data):
		content = data.strip().split(' ')
		if len(content) < 2:
			return None, None
		try:
			label = float(content[0])
			if label > 0.0:
				label = 1.0
			feas = {}
			for item in content[1:]:
				k, v = item.split(':')
				feas[k] = float(v)
		except:
			return None, None
		return label, feas

	def _load_data(self, fin):
		label_list = []
		feas_list = []
		if fin == None:
			return label_list, feas_list
		for line in fin:
			label, feas = self._parse_input(line)
			label_list.append(label)
			feas_list.append(feas)
		return label_list, feas_list

	def _train(self, label, feas):
		if self.opt == 'ftrl':
			return self._ftrl_update(label, feas)
		else:
			return self._sgd_update(label, feas, per_coordinate_lr=True)

	def _predict(self, feas):
		logit = 0.0
		for k, v in feas.items():
			if v == 0.0:
				continue
			logit += self.w[k] * v
		return sigmoid(logit)

	def _sparsity(self):
		zeros = 0
		for v in self.w.values():
			if v == 0:
				zeros += 1
		return float(zeros) / len(self.w), zeros, len(self.w)

	def val(self):
		preds = []
		for i in range(len(self.val_label_list)):
			preds.append(self._predict(self.val_feas_list[i]))
		auc = roc_auc_score(self.val_label_list, preds)
		return auc

	def train(self):
		count = 0
		loss = 0.0
		labels = []
		preds = []
		for line in self.train_file:
			label, feas = self._parse_input(line)
			print(label, feas)
			if label == None:
				continue
			if self.with_bias:
				feas['#bias#'] = 1.0
			count += 1
			loss += self._train(label, feas)
			labels.append(label)
			preds.append(self._predict(feas))
			if count % self.log_size == 0:
				auc = roc_auc_score(labels, preds)
				val_auc = self.val()
				print(f"count: {count:0>9d}, train_loss: {loss/self.log_size:>7.4f}, train_auc: {auc:>7.4f}, val_auc: {val_auc:>7.4f}")
				sparsity, zero_wei_num, all_wei_num = self._sparsity()
				print(f"count: {count:0>9d}, sparsity: {sparsity:>7.4f}, zero_weight_num: {zero_wei_num:0>5d}, all_weight_num: {all_wei_num:0>5d}")
				loss = 0.0
				labels = []
				preds = []
		auc = roc_auc_score(labels, preds)
		val_auc = self.val()
		print(f"count: {count:0>9d}, train_loss: {loss/self.log_size:>7.4f}, train_auc: {auc:>7.4f}, val_auc: {val_auc:>7.4f}")
		sparsity, zero_wei_num, all_wei_num = self._sparsity()
		print(f"count: {count:0>9d}, sparsity: {sparsity:>7.4f}, zero_weight_num: {zero_wei_num:0>5d}, all_weight_num: {all_wei_num:0>5d}")

if __name__ == '__main__':
	val_file = open('val_data', 'r')
	lr = LR(sys.stdin, val_file=val_file, lambda1=0.1)
	val_file.close()
	lr.train()
