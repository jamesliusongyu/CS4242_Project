import argparse
import torch
import torch.backends.cudnn as cudnn

import model
import evaluate
import data_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", default='[10, 20, 50, 100]', help="compute metrics@top_k")
    parser.add_argument("--data_path", type=str, default="../data/", help="main path for dataset")
    parser.add_argument("--model", type=str, default="MF", help="model name")
    parser.add_argument("--ckpt", type=str, default="MF_0.001lr_64emb_log.pth")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    cudnn.benchmark = True

    ############################## PREPARE DATASET ##########################
    train_path = args.data_path + '/training_dict.npy'
    valid_path = args.data_path + '/validation_dict.npy'
    test_path = args.data_path + '/testing_dict.npy'
    # test_path = args.data_path + '/heldout_dict.npy' # for live evaluation
    user_num, item_num, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt = data_utils.load_all(train_path, valid_path, test_path)

    ########################### LOAD MODEL #################################
    model = torch.load(f"./models/{args.ckpt}")
    model.to(args.device)

    ########################### EVALUATION #####################################
    model.eval()
    test_result = evaluate.metrics(args, model, eval(args.top_k), train_dict, test_dict, valid_dict, item_num, 1)

    print('---'*18)
    evaluate.print_results(None, None, test_result) 
    print('---'*18)