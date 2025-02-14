import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import model
import evaluate
import data_utils

import random
import numpy as np

if __name__ == "__main__":
	seed = 4242
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	cudnn.benchmark = True

	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", type=str, default="../data/", help="path for dataset")
	parser.add_argument("--model", type=str, default="MF", help="model name")
	parser.add_argument("--emb_size", type=int,default=64, help="predictive factors numbers in the model")

	# can play with learning rate and dropout
	parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
	parser.add_argument("--dropout", type=float,default=0.0,  help="dropout rate")
	parser.add_argument("--batch_size", type=int, default=1024, help="batch size for training")
	parser.add_argument("--epochs", type=int, default=100, help="training epoches")
	parser.add_argument("--device", type=str, default="cpu")

	parser.add_argument("--top_k", default='[10, 20, 50, 100]', help="compute metrics@top_k")
	parser.add_argument("--log_name", type=str, default='log', help="log_name")
	parser.add_argument("--model_path", type=str, default="./models/", help="main path for model")

	args = parser.parse_args()

	############################ PREPARE DATASET ##########################
	train_path = args.data_path + '/training_dict.npy'
	valid_path = args.data_path + '/validation_dict.npy'
	test_path = args.data_path + '/testing_dict.npy'
	user_num, item_num, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt = data_utils.load_all(train_path, valid_path, test_path)

	# construct the train datasets & dataloader
	train_dataset = data_utils.MFData(train_data, item_num, train_dict, True)
	train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

	########################### CREATE MODEL ##############################
	if args.model == 'MF':
		model = model.MF(user_num, item_num, args.emb_size, args.dropout)
	model.to(args.device)
	loss_function = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	########################### TRAINING ##################################
	best_recall = 0
	for epoch in range(args.epochs):
		# train
		model.train() # Enable dropout (if have).
		start_time = time.time()
		train_loader.dataset.ng_sample()

		for user, item, label in train_loader:
			user = user.to(args.device)
			item = item.to(args.device)
			label = label.float().to(args.device)

			model.zero_grad()

			prediction = model(user, item)
			loss = loss_function(prediction, label)

			loss.backward()
			optimizer.step()
		
		if (epoch+1) % 1 == 0:
			# evaluation
			model.eval()
			valid_result = evaluate.metrics(args, model, eval(args.top_k), train_dict, valid_dict, valid_dict, item_num, 0)
			test_result = evaluate.metrics(args, model, eval(args.top_k), train_dict, test_dict, valid_dict, item_num, 1)
			elapsed_time = time.time() - start_time

			print('---'*18)
			print("The time elapse of epoch {:03d}".format(epoch) + " is: " +  time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
			evaluate.print_results(None,valid_result,test_result)
			print('---'*18)

			if valid_result[0][0] > best_recall: # use best recall@10 on validation set to select the best results
				best_epoch = epoch
				best_recall = valid_result[0][0]
				best_results = valid_result
				best_test_results = test_result
				# save model
				if not os.path.exists(args.model_path):
					os.mkdir(args.model_path)
				torch.save(model, '{}{}_{}lr_{}emb_{}.pth'.format(args.model_path, args.model, args.lr, args.emb_size, args.log_name))
				
	print('==='*18)
	print(f"End. Best Epoch is {best_epoch}")
	evaluate.print_results(None,best_results,best_test_results)