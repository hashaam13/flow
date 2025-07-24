"""
Adapted from https://github.com/HannesStark/dirichlet-flow-matching/tree/main/model
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def count_params(model):
    return sum([p.numel() for p in model.parameters()])


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]


class CNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_cnn_stacks,p_dropout,):
        super().__init__()
        self.alphabet_size = vocab_size
        #self.num_cls = cfg.data.num_classes
        #self.classifier = cfg.model.classifier
        #if "cls_free_guidance" in cfg.model:
        #    self.cls_free_guidance = cfg.model.cls_free_guidance
        #else:
        #    self.cls_free_guidance = False
        hidden_dim = hidden_dim
        num_cnn_stacks = num_cnn_stacks
        p_dropout = p_dropout
        self.clean_data = False
        self.cls_free_guidance = False
        self.classifier= False
        if self.clean_data:
            self.linear = nn.Embedding(
                self.alphabet_size, embedding_dim=cfg.model.hidden_dim
            )
        else:
            inp_size = self.alphabet_size
            self.linear = nn.Conv1d(inp_size, hidden_dim, kernel_size=9, padding=4)
            self.time_embedder = nn.Sequential(
                GaussianFourierProjection(embed_dim=hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.num_layers = 5 * num_cnn_stacks
        self.convs = [
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, dilation=64, padding=256),
        ]
        self.convs = nn.ModuleList(
            [
                copy.deepcopy(layer)
                for layer in self.convs
                for i in range(num_cnn_stacks)
            ]
        )
        self.time_layers = nn.ModuleList(
            [Dense(hidden_dim, hidden_dim) for _ in range(self.num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)]
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(
                hidden_dim,
                hidden_dim if self.classifier else self.alphabet_size,
                kernel_size=1,
            ),
        )
        self.dropout = nn.Dropout(p_dropout)
        # If the model is built to be a classifier
        # Add a classification head on top
        if self.classifier:
            self.cls_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_cls),
            )

        if self.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(
                num_embeddings=self.num_cls + 1, embedding_dim=hidden_dim
            )
            self.cls_layers = nn.ModuleList(
                [Dense(hidden_dim, hidden_dim) for _ in range(self.num_layers)]
            )

    def forward(self, x, t, return_embedding=False):
        """
        Args:
            batch_data(dict):
                x: Input sequence, shape (B, D)
            t: Input time, shape (B,)
        """
        seq = x

        if self.clean_data:
            feat = self.linear(seq)
            feat = feat.permute(0, 2, 1)
        else:
            if len(seq.shape) == 3:
                # Assume already one-hot encoded
                seq_encoded = seq
            else:
                # Shape (B, D, S)
                seq_encoded = F.one_hot(
                    seq.long(), num_classes=self.alphabet_size
                ).float()
            time_emb = F.relu(self.time_embedder(t))
            feat = seq_encoded.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))


        # Input shape (B, S, D)
        for i in range(self.num_layers):
            h = self.dropout(feat.clone())

            if not self.clean_data:
                h = h + self.time_layers[i](time_emb)[:, :, None]

            if self.cls_free_guidance and not self.classifier:
                h = h + self.cls_layers[i](cls_emb)[:, :, None]

            h = self.norms[i]((h).permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h

        feat = self.final_conv(feat)
        feat = feat.permute(0, 2, 1)
        if self.classifier:
            feat = feat.mean(dim=1)
            if return_embedding:
                embedding = self.cls_head[:1](feat)
                return self.cls_head[1:](embedding), embedding
            else:
                return self.cls_head(feat)

        return feat