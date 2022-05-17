import sys
import torch
import torch.nn as nn
from sklearn import datasets

digits = datasets.load_digits()	
data = torch.tensor(digits.data).float() / 255
target = torch.tensor(digits.target)

train_data = data[:1000]
train_target = target[:1000]

val_data = data[1000:]
val_target = target[1000:]

def l1_loss(model):
	loss = 0
	for param in model.parameters():
		loss += torch.sum(torch.abs(param))
	return loss

def l2_loss(model):
	loss = 0
	for param in model.parameters():
		loss += torch.sum(param * param)
	return loss

def val(model):
	pred = model(val_data)
	return torch.sum(torch.argmax(pred, dim=1)==val_target) / float(val_target.shape[0])

def train(epoch=1, batch_size=1, l1=0, l2=0):
	size = train_data.shape[0]
	model = nn.Linear(64, 10)
	loss_fn = nn.CrossEntropyLoss()
	optim = torch.optim.SGD(model.parameters(), lr=0.01)

	for e in range(epoch):
		all_loss = 0
		all_f_loss = 0
		all_r_l1_loss = 0
		all_r_l2_loss = 0
		for i in range(0, size-batch_size, batch_size):
			X = train_data[i:i+batch_size]
			y = train_target[i:i+batch_size]
			pred = model(X)

			f_loss = loss_fn(pred, y)
			r_l1_loss = l1_loss(model)
			r_l2_loss = l2_loss(model)
			loss = f_loss + l1*r_l1_loss + l2*r_l2_loss

			all_loss += loss
			all_f_loss += f_loss
			all_r_l1_loss += r_l1_loss
			all_r_l2_loss += r_l2_loss

			optim.zero_grad()
			loss.backward()
			optim.step()
		acc = val(model)
		all_loss /= size
		all_f_loss /= size
		all_r_l1_loss /= size
		all_r_l2_loss /= size
		if e % 100 == 0 or e == epoch-1:
			print(f"epoch: {e:0>3d}, loss: {all_loss:>7.4f}, f_loss: {all_f_loss:>7.4f}, "
				+ f"l1_loss: {all_r_l1_loss:>7.4f}, l2_loss: {all_r_l2_loss:>7.4f}, acc: {acc:>7.4f}")
	
	for param in model.parameters():
		print(param)


if __name__ == '__main__':
	#train(epoch=1000, batch_size=32)
	#train(epoch=1000, batch_size=32, l2=0.005)
	train(epoch=1000, batch_size=32, l1=0.001)
