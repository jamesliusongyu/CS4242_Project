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
    RECALL, NDCG, ILD, F1 = [], [], [], []
    recommends = evaluate(args, model, top_k, train_dict, gt_dict,
                          valid_dict, item_num, flag, item_visual_map, item_category_map)

    for idx in range(len(top_k)):
        sumForRecall, sumForNDCG, user_length = 0, 0, 0
        k = -1
        per_user_recs = []

        for user_id in gt_dict.keys():
            k += 1
            if len(gt_dict[user_id]) != 0:
                userhit = 0
                dcg = 0
                idcg = 0
                idcgCount = len(gt_dict[user_id])
                ndcg = 0
                recs = recommends[idx][k]
                per_user_recs.append(recs)

                for index, thing in enumerate(recs):
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
        NDCG_score = round(sumForNDCG / user_length, 4)
        NDCG.append(NDCG_score)

        # Compute ILD and F1
        ILD_score = compute_ILD(per_user_recs, item_category_map, K=10)
        ILD.append(ILD_score)
        F1_score = compute_F1(NDCG_score, ILD_score)
        F1.append(F1_score)

    return RECALL, NDCG, ILD, F1


def sigmoid(x, mu=0.9895952383, sigma=0.0050332415):
    """ Sigmoid normalization for ILD. """
    return 1 / (1 + np.exp(-(x - mu) / sigma))


def compute_ILD(recommends, item_category_map, K=10):
    """ Compute Intra-List-Diversity@10 (ILD) for each user. """
    ILD_scores = []

    for user_recs in recommends:
        if len(user_recs) < 2:
            ILD_scores.append(0.0)  # No diversity if only one or no items

        diversity_sum = 0
        count = 0

        for i in range(min(K, len(user_recs))):
            for j in range(i + 1, min(K, len(user_recs))):
                if item_category_map.get(user_recs[i]) != item_category_map.get(user_recs[j]):
                    diversity_sum += 1
                count += 1

        if count > 0:
            ILD = (2 / (K * (K - 1))) * diversity_sum
        else:
            ILD = 0.0

        ILD_scores.append(sigmoid(ILD))  # Normalize using sigmoid

    return np.mean(ILD_scores)  # Average ILD across users


def compute_F1(NDCG, ILD):
    """ Compute F1 Score (NDCG-ILD) for the recommendation list. """
    if NDCG + ILD == 0:
        return 0.0  # Avoid division by zero
    return (2 * NDCG * ILD) / (NDCG + ILD)


def print_results(loss, valid_result, test_result):
    """Output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None:
        print("[Valid]: Recall: {} NDCG: {} ILD: {} F1: {}".format(
            '-'.join([str(x) for x in valid_result[0]]),
            '-'.join([str(x) for x in valid_result[1]]),
            '-'.join([str(x) for x in valid_result[2]]),
            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None:
        print("[Test]: Recall: {} NDCG: {} ILD: {} F1: {}".format(
            '-'.join([str(x) for x in test_result[0]]),
            '-'.join([str(x) for x in test_result[1]]),
            '-'.join([str(x) for x in test_result[2]]),
            '-'.join([str(x) for x in test_result[3]])))
