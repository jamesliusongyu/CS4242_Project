import numpy as np
import torch
import torch.utils.data as data


def load_all(train_path, valid_path, test_path, visual_path, category_path):
    """ We load all the three file here to save time in each epoch. """
    train_dict = np.load(train_path, allow_pickle=True).item()
    valid_dict = np.load(valid_path, allow_pickle=True).item()
    test_dict = np.load(test_path, allow_pickle=True).item()

    # Load visual and category features
    visual_features = np.load(visual_path, allow_pickle=True).item()
    category_features = np.load(category_path, allow_pickle=True).item()

    # Convert visual features to a matrix for normalization
    item_ids = list(visual_features.keys())
    visual_matrix = np.array([visual_features[item]
                             for item in item_ids], dtype=np.float32)  # (num_items, 512)

    # Compute mean and std across all items
    mean_visual = visual_matrix.mean(axis=0, keepdims=True)  # (1, 512)
    std_visual = visual_matrix.std(
        axis=0, keepdims=True) + 1e-8  # Avoid division by zero

    # Normalize visual features
    normalized_visual_matrix = (visual_matrix - mean_visual) / std_visual

    # Ensure correct user_num and item_num
    user_num = max(max(train_dict.keys(), default=0),
                   max(valid_dict.keys(), default=0),
                   max(test_dict.keys(), default=0)) + 1

    train_data, valid_gt, test_gt = [], [], []
    item_visual_map, item_category_map = {}, {}
    item_num = 0

    for user, items in train_dict.items():
        item_num = max(item_num, max(items, default=0))
        for item in items:
            train_data.append([int(user), int(item)])
            # Store normalized visual features
            item_visual_map[int(
                item)] = normalized_visual_matrix[item_ids.index(item)]
            # Store category ID (unchanged)
            item_category_map[int(item)] = int(
                category_features.get(int(item), -1))

    for user, items in valid_dict.items():
        item_num = max(item_num, max(items, default=0))
        for item in items:
            valid_gt.append([int(user), int(item)])

    for user, items in test_dict.items():
        item_num = max(item_num, max(items, default=0))
        for item in items:
            test_gt.append([int(user), int(item)])

    return (user_num, item_num + 1, train_dict, valid_dict, test_dict,
            train_data, valid_gt, test_gt, item_visual_map, item_category_map)


class MFData(data.Dataset):
    def __init__(self, features, num_item, item_visual_map, item_category_map, train_dict=None, is_training=False):
        super(MFData, self).__init__()
        """ Dataset for MF with content-based features (visual & category). """
        self.features_ps = features  # Positive user-item interactions
        self.num_item = num_item
        self.item_visual_map = item_visual_map  # Visual feature dictionary
        self.item_category_map = item_category_map  # Category dictionary
        self.train_dict = train_dict
        self.is_training = is_training
        self.labels = [1 for _ in range(len(features))]  # Positive labels (1)

        # Automatically generate negative samples during training
        if self.is_training:
            self.ng_sample()
        else:
            self.features_fill = self.features_ps
            self.labels_fill = self.labels
        print(
            f"[DEBUG] Initialized MFData: {len(features)} positive interactions loaded")

    def ng_sample(self):
        """ Negative sampling: randomly selects items that the user has NOT interacted with. """
        assert self.is_training, 'No need for negative sampling during testing'

        self.features_ng = []
        for user, _ in self.features_ps:
            neg_items = set()
            while len(neg_items) < 2:  # Generate 2 negatives per positive sample
                neg_item = np.random.randint(self.num_item)
                if neg_item not in self.train_dict[user]:
                    neg_items.add(neg_item)

            for neg_item in neg_items:
                self.features_ng.append([user, neg_item])

        self.labels_ng = [0 for _ in range(
            len(self.features_ng))]  # Negative labels (0)
        # Debugging: Print negative samples
        if len(self.features_ng) > 0:
            print(
                f"[DEBUG] Negative Sampling Example - User {self.features_ng[0][0]}: {self.features_ng[:5]}")

    # Merge
        # Merge positive & negative samples
        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = self.labels + self.labels_ng
        print(
            f"[DEBUG] Negative Sampling Done: {len(self.features_ng)} negatives generated")

    def __len__(self):
        return len(self.labels_fill)

    def __getitem__(self, idx):
        """ Returns (user, item, visual_feature, category_id, label) """
        features = self.features_fill
        labels = self.labels_fill

        user, item = features[idx]
        label = labels[idx]

        # Retrieve visual & category features
        visual_feature = torch.tensor(self.item_visual_map.get(
            item, np.zeros(512)), dtype=torch.float32)
        category_id = torch.tensor(self.item_category_map.get(
            item, -1), dtype=torch.long)  # Default to -1 if missing

        if idx < 5:  # Print only first few samples
            print(
                f"[DEBUG] Data Sample - User: {user}, Item: {item}, Category: {category_id}, Label: {label}")

        return user, item, visual_feature, category_id, label
