import torch
import torch.nn as nn
from .constants import ScoreFNType


class Module(nn.Module):
    def __init__(
        self,
        dim: int,
        score_fn_type: ScoreFNType='concat',
        dropout: float=0.2,
    ):
        super().__init__()

        self.dim = dim
        self.score_fn_type = score_fn_type
        self.dropout = dropout

        self._init_layers()

    def forward(self, Q, K):
        # Q: (B, 1, D)
        # K: (B, K, D)
        B, K_len, D = K.shape

        if self.score_fn_type == 'concat':
            Q_exp = Q.expand(-1, K_len, -1)  # (B, K, D)
            QK_cat = torch.cat([Q_exp, K], dim=-1)  # (B, K, 2D)
            scores = self.mlp(QK_cat).squeeze(-1)  # (B, K)
            return scores

        elif self.score_fn_type == 'hadamard':
            Q_exp = Q.expand(-1, K_len, -1)  # (B, K, D)
            QK_hadamard = Q_exp * K  # (B, K, D)
            scores = self.mlp(QK_hadamard).squeeze(-1)  # (B, K)
            return scores

    def _init_layers(self):
        if self.score_fn_type == 'concat':
            self.mlp = nn.Sequential(
                nn.Linear(self.dim * 2, self.dim),
                nn.LayerNorm(self.dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),

                nn.Linear(self.dim, 1),
            )
        elif self.score_fn_type == 'hadamard':
            self.mlp = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.LayerNorm(self.dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),

                nn.Linear(self.dim, 1),
            )
        else:
            raise ValueError("score_fn_type must be `concat` or `hadamard`")