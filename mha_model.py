import torch
import torch.nn as nn
from collections import OrderedDict
from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import PredictionLayer

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        x = x.transpose(0, 1)  # Shape: (seq_len, batch_size, embed_dim)
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = x + attn_output  # Residual connection
        attn_output = self.layernorm(attn_output)
        attn_output = attn_output.transpose(0, 1)  # Shape: (batch_size, seq_len, embed_dim)
        return attn_output

class MultiHeadAttentionModel(BaseModel):
    def __init__(self,
                 linear_feature_columns,
                 dnn_feature_columns,
                 num_heads=4,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=1e-5,
                 l2_reg_embedding=1e-5,
                 l2_reg_dnn=0,
                 dnn_dropout=0,
                 init_std=0.0001,
                 seed=1024,
                 task='binary',
                 device='cpu',
                 gpus=None):
        
        super(MultiHeadAttentionModel, self).__init__(linear_feature_columns, dnn_feature_columns,
                                                      l2_reg_linear=l2_reg_linear,
                                                      l2_reg_embedding=l2_reg_embedding,
                                                      init_std=init_std, seed=seed,
                                                      task=task, device=device, gpus=gpus)
        
        self.num_heads = num_heads
        self.dnn_hidden_units = dnn_hidden_units
        self.task = task
        self.device = device

        # Separate sparse and dense feature columns
        self.sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), self.dnn_feature_columns))
        self.dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), self.dnn_feature_columns))

        # Embedding dimension
        self.att_embedding_size = self.embedding_size  # Should be equal to embedding_dim

        # Multi-Head Self-Attention Layer
        self.multihead_attention = MultiHeadSelfAttention(self.att_embedding_size, num_heads)
        
        # Project dense features into embedding space
        if len(self.dense_feature_columns) > 0:
            num_dense_features = len(self.dense_feature_columns)
            self.dense_projection = nn.Linear(num_dense_features, num_dense_features * self.att_embedding_size)
        else:
            self.dense_projection = None

        # DNN Layer
        dnn_input_dim = self.att_embedding_size * (len(self.sparse_feature_columns) + len(self.dense_feature_columns))
        if len(dnn_hidden_units) > 0:
            dnn_layers = []
            input_dim = dnn_input_dim
            for i, unit in enumerate(dnn_hidden_units):
                dnn_layers.append(('linear{}'.format(i), nn.Linear(input_dim, unit)))
                dnn_layers.append(('batchnorm{}'.format(i), nn.BatchNorm1d(unit)))
                dnn_layers.append(('activation{}'.format(i), nn.ReLU()))
                if dnn_dropout > 0:
                    dnn_layers.append(('dropout{}'.format(i), nn.Dropout(p=dnn_dropout)))
                input_dim = unit
            self.dnn = nn.Sequential(OrderedDict(dnn_layers))
            self.dnn_linear = nn.Linear(input_dim, 1)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()),
                l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        else:
            self.dnn = None

        self.out = PredictionLayer(task)
        self.to(device)

    def forward(self, X):
        # Linear Part
        linear_logit = self.linear_model(X)

        # Embedding Layer
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        # Process Sparse Embeddings
        sparse_embedding_list = [emb.squeeze(1) for emb in sparse_embedding_list]
        sparse_embeddings = torch.stack(sparse_embedding_list, dim=1)  # Shape: (batch_size, num_sparse_features, embed_dim)

        # Process Dense Features
        if len(dense_value_list) > 0:
            dense_values = torch.cat(dense_value_list, dim=1)  # Shape: (batch_size, num_dense_features)
            # Project dense features into embedding space
            projected_dense = self.dense_projection(dense_values)  # Shape: (batch_size, num_dense_features * embed_dim)
            batch_size = projected_dense.size(0)
            num_dense_features = len(self.dense_feature_columns)
            projected_dense = projected_dense.view(batch_size, num_dense_features, self.att_embedding_size)  # Shape: (batch_size, num_dense_features, embed_dim)
            # Combine Sparse and Dense Embeddings
            embeddings = torch.cat([sparse_embeddings, projected_dense], dim=1)  # Shape: (batch_size, num_fields, embed_dim)
        else:
            embeddings = sparse_embeddings

        # Multi-Head Attention
        attn_output = self.multihead_attention(embeddings)  # Shape: (batch_size, num_fields, embed_dim)

        # Flatten the attention output
        attn_output_flat = attn_output.contiguous().view(attn_output.size(0), -1)  # Shape: (batch_size, num_fields * embed_dim)


        if self.dnn is not None:
            dnn_input = attn_output_flat
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit = linear_logit + dnn_logit
        else:
            logit = linear_logit

        y_pred = self.out(logit)
        return y_pred
