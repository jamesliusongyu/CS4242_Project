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
from torch.optim.lr_scheduler import ReduceLROnPlateau

if __name__ == "__main__":
    seed = 4242
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="../data/", help="path for dataset")
    parser.add_argument("--model", type=str, default="MF", help="model name")
    parser.add_argument("--emb_size", type=int, default=128,
                        help="predictive factors numbers in the model")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--dropout", type=float,
                        default=0.3, help="dropout rate")
    parser.add_argument("--batch_size", type=int,
                        default=1024, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="training epochs")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--top_k", default='[10, 20, 50]', help="compute metrics@top_k")
    parser.add_argument("--log_name", type=str, default='log', help="log_name")
    parser.add_argument("--model_path", type=str,
                        default="./models/", help="main path for model")
    args = parser.parse_args()

    ############################ PREPARE DATASET ##########################
    train_path = args.data_path + '/training_dict.npy'
    valid_path = args.data_path + '/validation_dict.npy'
    test_path = args.data_path + '/testing_dict.npy'
    visual_path = args.data_path + '/visual_feature.npy'
    category_path = args.data_path + '/category_feature.npy'

    user_num, item_num, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt, item_visual_map, item_category_map = data_utils.load_all(
        train_path, valid_path, test_path, visual_path, category_path)

    # Print first 10 training interactions
    print("\n🔍 First 10 Training Interactions:")
    for i, (user, item) in enumerate(train_data[:10]):
        print(f"  - User {user} interacted with Item {item}")
    print("=" * 60)

    # Print first 10 validation interactions
    print("\n🔍 First 10 Validation Interactions:")
    for i, (user, item) in enumerate(valid_gt[:10]):
        print(f"  - User {user} interacted with Item {item}")
    print("=" * 60)

    # Print first 10 test interactions
    print("\n🔍 First 10 Test Interactions:")
    for i, (user, item) in enumerate(test_gt[:10]):
        print(f"  - User {user} interacted with Item {item}")
    print("=" * 60)

    # Print first 10 users and their interacted items in train_dict
    print("\n🔍 First 10 Users and their Interactions (Training Set):")
    for user_id in list(train_dict.keys())[:10]:
        print(f"  - User {user_id}: {train_dict[user_id]}")
    print("=" * 60)

    # Print first 10 users and their interacted items in valid_dict
    print("\n🔍 First 10 Users and their Interactions (Validation Set):")
    for user_id in list(valid_dict.keys())[:10]:
        print(f"  - User {user_id}: {valid_dict[user_id]}")
    print("=" * 60)

    # Print first 10 users and their interacted items in test_dict
    print("\n🔍 First 10 Users and their Interactions (Test Set):")
    for user_id in list(test_dict.keys())[:10]:
        print(f"  - User {user_id}: {test_dict[user_id]}")
    print("=" * 60)

    # Print first 10 items and their visual features
    print("\n🖼️ First 10 Items with Visual Features:")
    for item_id in list(item_visual_map.keys())[:10]:
        # Print first 5 values for brevity
        print(f"  - Item {item_id}: {item_visual_map[item_id][:5]} ...")
    print("=" * 60)

    # Print first 10 items and their category features
    print("\n📊 First 10 Items with Category Features:")
    for item_id in list(item_category_map.keys())[:10]:
        print(f"  - Item {item_id}: Category {item_category_map[item_id]}")
    print("=" * 60)

    # construct the train datasets & dataloader
    train_dataset = data_utils.MFData(
        train_data, item_num, item_visual_map, item_category_map, train_dict, True)
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    ########################### CREATE MODEL ##############################
    num_categories = max(item_category_map.values()) + \
        1  # Get the highest category ID + 1
    if args.model == 'MF':
        print(user_num, item_num, args.emb_size, num_categories)
        model = model.ContentFM(user_num, item_num, args.emb_size,
                                num_categories, visual_dim=512, dropout=args.dropout)

    model.to(args.device)
    loss_function = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Track losses
    train_losses = []
    val_losses = []
    best_recall = 0

    def bpr_loss(pos_pred, neg_pred):
        """ Bayesian Personalized Ranking (BPR) Loss """
        return -torch.mean(torch.log(torch.sigmoid(pos_pred - neg_pred)))

    ########################### TRAINING ##################################
    best_recall = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        # 🔹 Generate Negative Samples
        train_loader.dataset.ng_sample()

        for batch_idx, (user, item, visual_feat, category_id, label) in enumerate(train_loader):
            user = user.to(args.device)
            item = item.to(args.device)
            label = label.float().to(args.device)

            # Convert visual & category features
            visual_feat = torch.tensor(
                [item_visual_map[int(i)] for i in item.cpu().numpy()], dtype=torch.float32
            ).to(args.device)
            category_id = torch.tensor(
                [item_category_map[int(i)] for i in item.cpu().numpy()], dtype=torch.long
            ).to(args.device)

            model.zero_grad()
            prediction = model(user, item, visual_feat, category_id)
            loss = loss_function(prediction, label)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"[Epoch {epoch+1} | Batch {batch_idx}] Loss: {loss.item():.6f}")

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 🔹 Step the scheduler
        scheduler.step(avg_train_loss)
        print(
            f"Epoch {epoch+1}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        # 🔹 Evaluation
        if (epoch + 1) % 1 == 0:
            model.eval()
            valid_result = evaluate.metrics(args, model, eval(
                args.top_k), train_dict, valid_dict, valid_dict, item_num, 0, item_visual_map, item_category_map)
            test_result = evaluate.metrics(args, model, eval(
                args.top_k), train_dict, test_dict, valid_dict, item_num, 1, item_visual_map, item_category_map)
            elapsed_time = time.time() - start_time

            print("=" * 60)
            print(f"✅ Epoch {epoch+1} Complete! Total Loss: {total_loss:.6f}")
            print(
                f"⏳ Time Taken: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
            evaluate.print_results(None, valid_result, test_result)
            print("=" * 60)

            # 🔹 Save Best Model Based on Recall@10
            if valid_result[0][0] > best_recall:
                best_epoch = epoch
                best_recall = valid_result[0][0]
                best_results = valid_result
                best_test_results = test_result

                if not os.path.exists(args.model_path):
                    os.makedirs(args.model_path)

                torch.save(model, '{}{}_{}lr_{}emb_{}.pth'.format(
                    args.model_path, args.model, args.lr, args.emb_size, args.log_name))

    # 🔹 Final Logging
    print("=" * 60)
    print(f"🎉 Training Complete! Best Model Found at Epoch {best_epoch+1}")
    evaluate.print_results(None, best_results, best_test_results)
    print("=" * 60)
