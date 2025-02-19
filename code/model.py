import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentFM(nn.Module):
    def __init__(self, num_users, num_items, emb_size=128, num_categories=100, visual_dim=512, dropout=0.2):
        super(ContentFM, self).__init__()

        # 🔹 User and Item Embeddings
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)

        # 🔹 Content Feature Embeddings (Category & Visual)
        self.category_emb = nn.Embedding(
            num_categories, emb_size)  # Category Feature Embedding
        # Linear transformation for visual features
        self.visual_emb = nn.Linear(visual_dim, emb_size)

        # 🔹 Bias Terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # 🔹 Factorization Machine (FM) Linear Weights
        # 4 embeddings: user, item, category, visual
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

        # 🔹 Initialize Weights
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.category_emb.weight)
        nn.init.xavier_uniform_(self.visual_emb.weight)

        self.dropout = nn.Dropout(dropout)

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

        return prediction
