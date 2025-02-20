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
import matplotlib.pyplot as plt
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
    parser.add_argument("--emb_size", type=int,
                        default=128, help="embedding size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--dropout", type=float,
                        default=0.4, help="dropout rate")
    parser.add_argument("--batch_size", type=int,
                        default=1024, help="batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="training epochs")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--top_k", default='[10, 20, 50]', help="compute metrics@top_k")
    parser.add_argument("--log_name", type=str, default='log', help="log name")
    parser.add_argument("--model_path", type=str,
                        default="./models/", help="model save path")
    args = parser.parse_args()

    # Load dataset
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
    print(len(train_data))
    print("=" * 60)

    # Print first 10 validation interactions
    print("\n🔍 First 10 Validation Interactions:")
    for i, (user, item) in enumerate(valid_gt[:10]):
        print(f"  - User {user} interacted with Item {item}")
    print(len(valid_gt))
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
    val_dataset = data_utils.MFData(
        valid_gt, item_num, item_visual_map, item_category_map, valid_dict, False)
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)  # ✅ No shuffle

    # Initialize model
    num_categories = max(item_category_map.values()) + 1
    if args.model == 'MF':
        model = model.ContentFM(user_num, item_num, num_categories, args.emb_size,
                                visual_dim=512, dropout=args.dropout)

    model.to(args.device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Early Stopping Parameters
    patience = 100  # Stop training if validation loss doesn't improve for 5 epochs
    patience_count = 0
    best_valid_loss = float('inf')

    # Track losses
    train_losses = []
    val_losses = []
    best_recall = 0

    # Training Loop with Early Stopping
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        train_loader.dataset.ng_sample()  # Generate negative samples

        for batch_idx, (user, item, visual_feat, category_id, label) in enumerate(train_loader):
            user = user.to(args.device)
            item = item.to(args.device)
            label = label.float().to(args.device)
            # print(label, "label")
            # tensor([0., 1., 0.,  ..., 0., 0., 0.]) label
            # this means there there are 1024 items, and 1 indicates yes, 0 indicates no
            visual_feat = torch.tensor([item_visual_map.get(int(i), np.zeros(512)) for i in item.cpu().numpy()],
                                       dtype=torch.float32).to(args.device)
            category_id = torch.tensor([item_category_map.get(int(i), 0) for i in item.cpu().numpy()],
                                       dtype=torch.long).to(args.device)

            optimizer.zero_grad()
            prediction = model(user, item, visual_feat, category_id)
            loss = loss_function(prediction, label)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Step scheduler
        scheduler.step(avg_train_loss)
        print(
            f"Epoch {epoch+1}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Compute Validation Loss
        valid_loss = 0

        with torch.no_grad():
            for user, item, visual_feat, category_id, label in val_loader:
                user = user.to(args.device)
                item = item.to(args.device)
                label = label.float().to(args.device)

                visual_feat = torch.tensor([item_visual_map.get(int(i), np.zeros(512)) for i in item.cpu().numpy()],
                                           dtype=torch.float32).to(args.device)
                category_id = torch.tensor([item_category_map.get(int(i), 0) for i in item.cpu().numpy()],
                                           dtype=torch.long).to(args.device)

                prediction = model(user, item, visual_feat, category_id)
                loss = loss_function(prediction, label)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(val_loader)
        val_losses.append(avg_valid_loss)

        if (epoch+1) % 1 == 0:
            # evaluation
            model.eval()
            valid_result = evaluate.metrics(args, model, eval(
                args.top_k), train_dict, valid_dict, valid_dict, item_num, 0, item_visual_map, item_category_map)
            test_result = evaluate.metrics(args, model, eval(
                args.top_k), train_dict, test_dict, valid_dict, item_num, 1, item_visual_map, item_category_map)
            elapsed_time = time.time() - start_time

            print('---'*18)
            print("The time elapse of epoch {:03d}".format(
                epoch) + " is: " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
            evaluate.print_results(None, valid_result, test_result)
            print('---'*18)

            # use best recall@10 on validation set to select the best results
            if valid_result[0][0] > best_recall:
                best_epoch = epoch
                best_recall = valid_result[0][0]
                best_results = valid_result
                best_test_results = test_result
                # save model
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                torch.save(model, '{}{}_{}lr_{}emb_{}.pth'.format(
                    args.model_path, args.model, args.lr, args.emb_size, args.log_name))

        # ✅ Early Stopping Logic
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_count = 0  # Reset patience since loss improved
            best_epoch = epoch

            # # Save Best Model
            # if not os.path.exists(args.model_path):
            #     os.makedirs(args.model_path)

            # torch.save(
            #     model, f'{args.model_path}{args.model}_{args.lr}lr_{args.emb_size}emb_{args.log_name}.pth')

        else:
            patience_count += 1  # Increase patience count if no improvement

        if patience_count >= patience:
            print(
                f"⏹️ Early stopping triggered at epoch {epoch+1}. No validation loss improvement for {patience} epochs.")
            break

        print(
            f"✅ Epoch {epoch+1} Complete! Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_valid_loss:.6f}")

    # ✅ Plot Training vs. Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses,
             label="Training Loss", marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses,
             label="Validation Loss", marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss with Early Stopping")
    plt.grid()
    plt.savefig("loss_plot.png")  # Save plot
    plt.show()

    # Final log
    print("=" * 60)
    print(f"🎉 Training Complete! Best Model Found at Epoch {best_epoch+1}")
    print("=" * 60)
