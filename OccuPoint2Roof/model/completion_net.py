import torch
import torch.nn as nn

class FCABlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, n, c = x.size()
        z = x.mean(dim=1)
        w = self.fc(z).view(b, 1, c)
        return x * w

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(3, d_model)

    def forward(self, coords):
        return self.fc(coords)

class OccludedCompletionNet(nn.Module):
    def __init__(self, feature_dim, num_queries=30):
        super().__init__()
        self.num_queries = num_queries

        self.query_embed = nn.Embedding(num_queries, feature_dim)

        self.query_anchors = nn.Embedding(num_queries, 3)
        nn.init.uniform_(self.query_anchors.weight, -0.5, 0.5)

        self.pos_embedding = PositionalEmbedding(feature_dim)

        self.transformer = nn.Transformer(
            d_model=feature_dim, nhead=4,
            num_encoder_layers=3, num_decoder_layers=3,
            dim_feedforward=feature_dim * 2, batch_first=True
        )

        self.coord_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 3)
        )

        self.valid_head = nn.Linear(feature_dim, 1)

    def forward(self, global_feature, visible_features, visible_coords):
        b = global_feature.shape[0]
        device = global_feature.device

        if len(global_feature.shape) == 2:
            global_feature = global_feature.unsqueeze(1)

        vis_pos_emb = self.pos_embedding(visible_coords)
        visible_features = visible_features + vis_pos_emb

        memory_input = torch.cat([global_feature, visible_features], dim=1)

        base_anchors = self.query_anchors.weight.unsqueeze(0).repeat(b, 1, 1)
        anchor_pos_emb = self.pos_embedding(base_anchors)
        queries = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        dynamic_queries = queries + anchor_pos_emb

        out_features = self.transformer(src=memory_input, tgt=dynamic_queries)

        pred_offsets = self.coord_head(out_features)
        pred_valid_logits = self.valid_head(out_features).squeeze(-1)
        pred_occ_coords = base_anchors + pred_offsets

        return pred_occ_coords, pred_valid_logits