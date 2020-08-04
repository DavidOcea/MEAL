import math
import numpy as np

def lr_cosine(cur_epoch,total_epoch,base_lr):
	return 0.5 * base_lr * (1 + math.cos(math.pi * cur_epoch / total_epoch))

def warmup_learning_rate(base_lr, epoch, iter, iters, warmup_iters):
	cur_iter = epoch * iters + iter
	lr = cur_iter * base_lr / warmup_iters
	return lr

def lr_MultiStepLR(cur_epoch,total_epoch,base_lr,MultiStep=[0,40,70], gamma=0.1):
	div = 0
	if cur_epoch > MultiStep[-1]:
		div = len(MultiStep) - 1
	else:
		for idx, v in enumerate(MultiStep):
			if cur_epoch > MultiStep[idx] and cur_epoch <= MultiStep[idx + 1]:
				div = idx
				break
	return base_lr * (gamma**div)

def adjust_learning_rate(optimizer, cur_epoch, total_epoch, base_lr, lr_type="cosine", warmup=0, iter=0, iters=0, steps=[0,40,70]):
	""" adjust learning of a given optimizer and return the new learning rate """
	if cur_epoch < warmup:
		warmup_iters = warmup * iters
		new_lr = warmup_learning_rate(base_lr, cur_epoch, iter, iters, warmup_iters)
	else:
		if lr_type == "cosine":
			new_lr = lr_cosine(cur_epoch, total_epoch, base_lr)
		elif lr_type == "MultiStepLR":
			new_lr = lr_MultiStepLR(cur_epoch, total_epoch, base_lr, steps)
		
	for param_group in optimizer.param_groups:
		param_group['lr'] = new_lr
	return new_lr


if __name__ == "__main__":
	import torchvision.models as models
	import torch
	import matplotlib.pyplot as plt

	model = models.__dict__['resnet18'](num_classes=1000)
	base_lr = 0.01
	optimizer = torch.optim.SGD(model.parameters(), base_lr,
								momentum=0.9,
								weight_decay=0.01,
								nesterov=True)
	epochs = 100

	lrs = []
	for e in range(epochs):
		lr = adjust_learning_rate(optimizer, e, epochs, base_lr,base_lr,0,1000, lr_type='cosine',warmup=5)
		lrs.append(lr)
	print(lrs)

	x = np.linspace(0, 1, 100)
	plt.figure()
	plt.plot(x,lrs)
	plt.savefig("easyplot.jpg")


