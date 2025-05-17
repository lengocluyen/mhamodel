import torch
import torch.nn as nn
from collections import OrderedDict
from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import PredictionLayer

class PositionalEncoding(nn.Module):
    def __init__(self, num_fields, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_fields, embed_dim))

    def forward(self, x):
        return x + self.pos_embed

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ModuleList([nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)])

    def forward(self, x0):
        x = x0
        for layer in self.weights:
            # cross: x0 * (w^T x) + b + x
            w = layer.weight.view(-1, 1)
            b = layer.bias
            x = x0 * (x @ w).expand_as(x0) + b + x
        return x

class MultiHeadAttentionModel(BaseModel):
    def __init__(
        self,
        linear_feature_columns,
        dnn_feature_columns,
        num_heads=4,
        dnn_hidden_units=(256, 128),
        embed_dim=8,
        dnn_dropout=0.5,
        l2_reg_linear=1e-5,
        l2_reg_embedding=1e-5,
        l2_reg_dnn=1e-3,
        task='regression',
        device='cpu',
        gpus=None
    ):
        super().__init__(
            linear_feature_columns,
            dnn_feature_columns,
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            task=task,
            device=device,
            gpus=gpus
        )
        self.embed_dim = self.embedding_size
        self.sparse_feats = [f for f in dnn_feature_columns if isinstance(f, SparseFeat)]
        self.dense_feats  = [f for f in dnn_feature_columns if isinstance(f, DenseFeat)]
        self.num_fields   = len(self.sparse_feats) + len(self.dense_feats)
        # CLS token + positional
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_enc = PositionalEncoding(self.num_fields+1, self.embed_dim)
        # Attention
        self.attn1 = MultiHeadSelfAttention(self.embed_dim, num_heads, dropout=dnn_dropout)
        self.attn2 = MultiHeadSelfAttention(self.embed_dim, num_heads, dropout=dnn_dropout)
        # Cross-attn for CLS
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, num_heads, dropout=dnn_dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(self.embed_dim)
        # Dense projection
        if self.dense_feats:
            self.dense_proj = nn.Linear(len(self.dense_feats), len(self.dense_feats)*self.embed_dim)
        else:
            self.dense_proj = None
        # Cross network on CLS
        self.cross_net = CrossNetwork(self.embed_dim, num_layers=2)
        # DNN head
        dnn_input = self.embed_dim
        layers = OrderedDict()
        input_dim = dnn_input
        for i, h in enumerate(dnn_hidden_units):
            layers[f'linear{i}'] = nn.Linear(input_dim, h)
            layers[f'norm{i}']   = nn.LayerNorm(h)
            layers[f'act{i}']    = nn.ReLU()
            layers[f'drop{i}']   = nn.Dropout(dnn_dropout)
            input_dim = h
        self.dnn = nn.Sequential(layers)
        self.dnn_out = nn.Linear(input_dim, 1)
        self.prediction = PredictionLayer(task)
        self.add_regularization_weight(self.dnn_out.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        linear_logit = self.linear_model(X)
        sparse_embs, dense_vals = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        fields = [e.squeeze(1).unsqueeze(1) for e in sparse_embs]
        if dense_vals:
            dv = torch.cat(dense_vals, dim=1)
            proj = self.dense_proj(dv).view(-1, len(self.dense_feats), self.embed_dim)
            fields.append(proj)
        emb = torch.cat(fields, dim=1)
        bs = emb.size(0)
        cls = self.cls_token.expand(bs, -1, -1)
        emb = torch.cat([cls, emb], dim=1)
        emb = self.pos_enc(emb)
        out = self.attn1(emb)
        out = self.attn2(out)
        cls_q = out[:, :1, :]
        cross, _ = self.cross_attn(cls_q, out, out)
        cls_upd = self.cross_norm(cls_q + cross)
        feat = cls_upd.squeeze(1)
        # cross network enhancement
        feat = self.cross_net(feat)
        x = self.dnn(feat)
        dnn_logit = self.dnn_out(x)
        logit = linear_logit + dnn_logit
        return self.prediction(logit)