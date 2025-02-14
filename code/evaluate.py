import numpy as np
import torch

def evaluate(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag):
	recommends = []
	for i in range(len(top_k)):
		recommends.append([])

	with torch.no_grad():
		pred_list_all = []
		for i in gt_dict.keys(): # for each user
			if len(gt_dict[i]) != 0: # if 
				user = torch.full((item_num,), i, dtype=torch.int64).to(args.device) # create n_item users for prediction
				item = torch.arange(0, item_num, dtype=torch.int64).to(args.device) 
				prediction = model(user, item)
				prediction = prediction.detach().cpu().numpy().tolist()
				for j in train_dict[i]: # mask train
					prediction[j] -= float('inf')
				if flag == 1: # mask validation
					if i in valid_dict:
						for j in valid_dict[i]:
							prediction[j] -= float('inf')
				pred_list_all.append(prediction)

		predictions = torch.Tensor(pred_list_all).to(args.device) # shape: (n_user,n_item)
		for idx in range(len(top_k)):
			_, indices = torch.topk(predictions, int(top_k[idx]))
			recommends[idx].extend(indices.tolist())
	return recommends

def metrics(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag):
	RECALL, NDCG = [], []
	recommends = evaluate(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag)

	for idx in range(len(top_k)):
		sumForRecall, sumForNDCG, user_length = 0, 0, 0
		k=-1
		for i in gt_dict.keys(): # for each user
			k += 1
			if len(gt_dict[i]) != 0:
				userhit = 0
				dcg = 0
				idcg = 0
				idcgCount = len(gt_dict[i])
				ndcg = 0

				for index, thing in enumerate(recommends[idx][k]):
					if thing in gt_dict[i]:
						userhit += 1
						dcg += 1.0 / (np.log2(index+2))
					if idcgCount > 0:
						idcg += 1.0 / (np.log2(index+2))
						idcgCount -= 1
				if (idcg != 0):
					ndcg += (dcg / idcg)

				sumForRecall += userhit / len(gt_dict[i])
				sumForNDCG += ndcg
				user_length += 1

		RECALL.append(round(sumForRecall/user_length, 4))
		NDCG.append(round(sumForNDCG/user_length, 4))

	return RECALL, NDCG

def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Recall: {} NDCG: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]])))
    if test_result is not None: 
        print("[Test]: Recall: {} NDCG: {} ".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]])))