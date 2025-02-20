import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn


class ContentFM(nn.Module):
    def __init__(self, num_users, num_items, num_categories, emb_size=128, visual_dim=512, dropout=0.2):
        super(ContentFM, self).__init__()
        # 🔹 User and Item Embeddings
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        # 🔹 Content Feature Embeddings (Category & Visual)
        self.category_emb = nn.Embedding(num_categories, emb_size)
        self.visual_emb = nn.Linear(
            visual_dim, emb_size, bias=False)  # No bias

        # 🔹 Bias Terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # 🔹 Factorization Machine (FM) Linear Weights
        self.fm_linear = nn.Linear(emb_size * 4, 1)

        # 🔹 Deep Component (MLP) to model higher-order feature interactions
        self.mlp = nn.Sequential(
            nn.Linear(emb_size * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        # 🔹 Initialize Weights Properly
        self.init_weights()

        self.dropout = nn.Dropout(dropout)

    def init_weights(self):
        """ Initialize all model weights properly """
        # Xavier Initialization for embeddings
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.category_emb.weight)
        nn.init.xavier_uniform_(self.visual_emb.weight)

        # Bias terms initialized to zero
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        # Xavier Initialization for Linear Layers
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.fm_linear.weight)
        nn.init.zeros_(self.fm_linear.bias)

    def forward(self, user_id, item_id, visual_feat, category_id):
        """
        Forward pass for Content-based FM with user, item, category, and visual features.
        """
        # 🔹 Get User & Item Embeddings
        user_emb = self.user_emb(user_id)
        item_emb = self.item_emb(item_id)

        # 🔹 Get Category & Visual Feature Embeddings
        category_emb = self.category_emb(category_id)

        visual_emb = self.visual_emb(visual_feat)
        # 🔹 Compute Bias Terms
        user_bias = self.user_bias(user_id).squeeze()
        item_bias = self.item_bias(item_id).squeeze()

        # 🔹 Concatenate Features
        concat_features = torch.cat(
            [user_emb, item_emb, category_emb, visual_emb], dim=1)

        # 🔹 FM Component: Linear Regression on concatenated features
        fm_output = self.fm_linear(concat_features).squeeze()

        # 🔹 MLP Component: Non-linear interactions between features
        mlp_output = self.mlp(concat_features).squeeze()

        # 🔹 Combine FM + MLP with Bias Terms
        prediction = fm_output + mlp_output + user_bias + item_bias

        if not self.training:
            prediction = torch.sigmoid(prediction)

        return prediction
