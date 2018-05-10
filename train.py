# -*- coding: utf-8 -*-
from __future__ import division

""" 
Trains a ResNeXt Model on Cifar10 and Cifar 100. Implementation as defined in:

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.

"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import argparse
import os
import json
from progressbar import ProgressBar
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from models.model import CifarResNeXt

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR',
	        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# Positional arguments
	parser.add_argument('data_path', type=str, help='Root for the Cifar dataset.')
	parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')
	parser.add_argument('--use_valid', type=bool, default=True, help='Use validation set.')
	parser.add_argument('--split_random_seed', type=int, default=1234, help='Random seed to randomly split train and validation.')
	# Optimization options
	parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')
	parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
	parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The Learning Rate.')
	parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
	parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
	parser.add_argument('--test_bs', type=int, default=10)
	parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
	                    help='Decrease learning rate at these epochs.')
	parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
	# Checkpoints
	parser.add_argument('--save', '-s', type=str, default='./', help='Folder to save checkpoints.')
	parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
	parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
	# Architecture
	parser.add_argument('--depth', type=int, default=29, help='Model depth.')
	parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
	parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
	parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
	# Acceleration
	parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
	parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
	# i/o
	parser.add_argument('--log', type=str, default='./', help='Log folder.')
	args = parser.parse_args()

	# Init logger
	if not os.path.isdir(args.log):
		os.makedirs(args.log)
	log = open(os.path.join(args.log, 'log.txt'), 'w')
	state = {k: v for k, v in args._get_kwargs()}
	log.write(json.dumps(state) + '\n')

	# Calculate number of epochs wrt batch size
	args.epochs = args.epochs * 128 // args.batch_size
	args.schedule = [x * 128 // args.batch_size for x in args.schedule]

	# Init dataset
	if not os.path.isdir(args.data_path):
		os.makedirs(args.data_path)

	mean = [x / 255 for x in [125.3, 123.0, 113.9]]
	std = [x / 255 for x in [63.0, 62.1, 66.7]]

	train_transform = transforms.Compose( [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
	                                       transforms.ToTensor(), transforms.Normalize(mean, std)])
	if args.use_valid:
		valid_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
	test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
	if args.dataset == 'cifar10':
		train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
		test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
		nlabels = 10
	else:
		train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
		test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
		nlabels = 100
	if args.use_valid:
		valid_ratio = 0.1
		n_train = int(len(train_data) * (1.0 - valid_ratio))
		shuffed_ind = range(n_train)
		np.random.RandomState(args.split_random_seed).shuffle(shuffed_ind)
		train_sampler = torch.utils.data.sampler.SubsetRandomSampler(shuffed_ind[:n_train])
		valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(shuffed_ind[n_train:])
		train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.prefetch, pin_memory=True)
		valid_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.prefetch, pin_memory=True)
		test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)
	else:
		n_train = len(train_data)
		train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
		test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)

	# Init checkpoints
	if not os.path.isdir(args.save):
		os.makedirs(args.save)

	# Init model, criterion, and optimizer
	net = CifarResNeXt(args.cardinality, args.depth, nlabels, args.base_width, args.widen_factor)
	print(net)
	if args.ngpu > 1:
		net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

	if args.ngpu > 0:
		net.cuda()

	optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
	                            weight_decay=state['decay'], nesterov=True)

	# train function (forward, backward, update)
	def train():
		net.train()
		loss_avg = 0.0
		bar = ProgressBar(max_value=len(train_loader))
		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())

			# forward
			output = net(data)

			# backward
			optimizer.zero_grad()
			loss = F.cross_entropy(output, target)
			loss.backward()
			optimizer.step()

			# exponential moving average
			loss_avg = loss_avg * 0.2 + float(loss) * 0.8

			bar.update(batch_idx)

		state['train_loss'] = loss_avg


	# test function (forward only)
	def test():
		net.eval()
		loss_avg = 0.0
		correct = 0
		eval_loader = valid_loader if args.use_valid else test_loader
		for batch_idx, (data, target) in enumerate(eval_loader):
			data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())

			# forward
			output = net(data)
			loss = F.cross_entropy(output, target)

			# accuracy
			pred = output.data.max(1)[1]
			correct += float(pred.eq(target.data).sum())

			# test loss average
			loss_avg += float(loss)

		if args.use_valid:
			state['valid_loss'] = loss_avg / len(eval_loader)
			state['valid_accuracy'] = correct / len(eval_loader.dataset)
		else:
			state['test_loss'] = loss_avg / len(eval_loader)
			state['test_accuracy'] = correct / len(eval_loader.dataset)


	# Main loop
	best_accuracy = 0.0
	for epoch in range(args.epochs):
		if epoch in args.schedule:
			state['learning_rate'] *= args.gamma
			for param_group in optimizer.param_groups:
				param_group['lr'] = state['learning_rate']

		state['epoch'] = epoch
		train()
		test()
		eval_accuracy = (state['valid_accuracy'] if args.use_valid else state['test_accuracy'])
		if eval_accuracy > best_accuracy:
			best_accuracy = eval_accuracy
			torch.save(net.state_dict(), os.path.join(args.save, 'model.pytorch'))
		log.write('%s\n' % json.dumps(state))
		log.flush()
		print(state)
		print("Best accuracy: %f" % best_accuracy)

	log.close()
