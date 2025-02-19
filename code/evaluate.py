import sys
import numpy as np
import torch
import torch.nn.functional as F


def evaluate(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag, item_visual_map, item_category_map):
    recommends = []
    for i in range(len(top_k)):
        recommends.append([])

    with torch.no_grad():
        pred_list_all = []
        for user_id in gt_dict.keys():  # for each user
            if len(gt_dict[user_id]) != 0:
                # Create user and item tensors
                # print("lalal")
                user = torch.full((item_num,), user_id,
                                  dtype=torch.int64).to(args.device)
                item = torch.arange(
                    0, item_num, dtype=torch.int64).to(args.device)
                # print(user, item, "user", "item")
                # print(f"[DEBUG] Type of received item_visual_map: {type(item_visual_map)}")  # Should be <class 'dict'>
                # Retrieve visual features and category IDs
                # Ensure visual_feat and category_id are the same size as item tensor
                visual_feat_np = np.array([item_visual_map.get(
                    int(i), np.zeros(512)) for i in item.cpu().numpy()])
                category_id_np = np.array(
                    [item_category_map.get(int(i), -1) for i in item.cpu().numpy()])

                # Convert to torch tensors
                visual_feat = torch.tensor(
                    visual_feat_np, dtype=torch.float32).to(args.device)
                category_id = torch.tensor(
                    category_id_np, dtype=torch.long).to(args.device)

                # Ensure they have the same batch size as `user` and `item`
                assert visual_feat.shape[0] == item.shape[
                    0], f"Mismatch: visual_feat {visual_feat.shape[0]} != item {item.shape[0]}"
                assert category_id.shape[0] == item.shape[
                    0], f"Mismatch: category_id {category_id.shape[0]} != item {item.shape[0]}"
                # Call the model with all necessary inputs
                prediction = model(user, item, visual_feat, category_id)
                # print(prediction.shape, "Shape")
                # print("asd", gt_dict.keys())

                prediction = prediction.detach().cpu().numpy().tolist()
                # print(prediction, "Prediction")
                # print(user.shape, item.shape, visual_feat.shape,
                #       category_id.shape)  # Check input shapes
                # print(prediction.shape)  # Check output shape before converting to list

                # Mask out training interactions
                for j in train_dict[user_id]:
                    prediction[j] -= float('inf')

                # Mask out validation interactions if flag == 1
                if flag == 1 and user_id in valid_dict:
                    for j in valid_dict[user_id]:
                        prediction[j] -= float('inf')

                pred_list_all.append(prediction)

        predictions = torch.Tensor(pred_list_all).to(
            args.device)  # shape: (n_user, n_item)

        for idx in range(len(top_k)):
            _, indices = torch.topk(predictions, int(top_k[idx]))
            recommends[idx].extend(indices.tolist())

    return recommends


def metrics(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag, item_visual_map, item_category_map):
    RECALL, NDCG = [], []
    recommends = evaluate(args, model, top_k, train_dict, gt_dict,
                          valid_dict, item_num, flag, item_visual_map, item_category_map)
    # print ("asdd", train_dict.keys())
    # print ("asdd", gt_dict.keys())
    # print ("asdd", valid_dict.keys())
    # print ("recommends",recommends[0][0])
    # print (len(recommends))
    # print (recommends[0])
    # print ('----------------')
    # print (recommends[1])
    # print (recommends[2])
    # print ('----------------')

    # print (recommends[3])

    # sys.exit(1)
    for idx in range(len(top_k)):
        sumForRecall, sumForNDCG, user_length = 0, 0, 0
        k = -1
        for user_id in gt_dict.keys():
            k += 1
            if len(gt_dict[user_id]) != 0:
                userhit = 0
                dcg = 0
                idcg = 0
                idcgCount = len(gt_dict[user_id])
                ndcg = 0

                for index, thing in enumerate(recommends[idx][k]):
                    if thing in gt_dict[user_id]:
                        userhit += 1
                        dcg += 1.0 / (np.log2(index + 2))
                    if idcgCount > 0:
                        idcg += 1.0 / (np.log2(index + 2))
                        idcgCount -= 1
                if idcg != 0:
                    ndcg += (dcg / idcg)

                sumForRecall += userhit / len(gt_dict[user_id])
                sumForNDCG += ndcg
                user_length += 1

        RECALL.append(round(sumForRecall / user_length, 4))
        NDCG.append(round(sumForNDCG / user_length, 4))

    return RECALL, NDCG


def compute_intra_list_diversity(model, test_gt, item_num, category_map, mu=0.9895952383, sigma=0.0050332415):
    """ Computes Intra-List Diversity@10 (ILD@10). """
    total_ild = 0
    for user in test_gt.keys():
        user_tensor = torch.tensor([user]).to(model.device)
        item_tensors = torch.tensor(list(range(item_num))).to(model.device)

        with torch.no_grad():
            scores = model(user_tensor, item_tensors)

        _, top_k_items = torch.topk(scores, k=10)
        top_k_items = top_k_items.cpu().numpy()

        ild = 0
        K = len(top_k_items)
        if K > 1:
            for i in range(K):
                for j in range(i + 1, K):
                    if category_map[top_k_items[i]] != category_map[top_k_items[j]]:
                        ild += 1
            ild /= (K * (K - 1))

        ild = F.sigmoid(torch.tensor((ild - mu) / sigma))  # Normalization
        total_ild += ild

    return total_ild / len(test_gt)


def compute_f1_score(ndcg, ild):
    """ Computes F1 score using NDCG and ILD. """
    if ndcg + ild == 0:
        return 0
    return 2 * (ndcg * ild) / (ndcg + ild)


def print_results(loss, valid_result, test_result):
    """Output the evaluation results."""
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
