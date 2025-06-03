from typing import Optional
import torch
import torch.nn as nn
from .attn_score_fn import Module as attn_score_fn
from .simplex_proj_fn import Module as simplex_proj_fn
from .constants import (
    ScoreFNType,
)


class Module(nn.Module):
    def __init__(
        self,
        dim: int,
        score_fn_type: ScoreFNType = 'concat',
        beta: float=0.5,
        dropout: float=0.2,
    ):
        super().__init__()
        self.dim = dim
        self.score_fn_type = score_fn_type
        self.beta = beta
        self.dropout = dropout

        self._init_layers()

    def forward(
        self,
        Q: torch.Tensor,  # (B, D)
        K: torch.Tensor,  # (B, K, D)
        V: torch.Tensor,  # (B, K, D)
        mask: Optional[torch.Tensor]=None,
    ):
        # (B, D) -> (B, 1, D)
        Q_exp = Q.unsqueeze(1)

        # Attention scores: (B, 1, K)
        scores = self.attn_score_fn(Q_exp, K)

        # Masking
        if mask is not None:
            scores = torch.masked_fill(
                input=scores,
                mask=self._match_dim(mask, scores),
                value=float('-inf'),
            )

        # Simplex projection
        # (B, K) -> (B, 1, K)
        weights = self.simplex_proj_fn(scores).unsqueeze(1)

        # Context vector: (B, D)
        # (B, 1, K) x (B, K, D) -> (B, 1, D) -> (B, D)
        context = torch.bmm(weights, V).squeeze(1)

        return context

    def _match_dim(self, source, target):
        if source is not None:
            while source.ndim < target.ndim:
                source = source.unsqueeze(1)
        return source

    def _init_layers(self):
        # Attention score function module
        self.attn_score_fn = attn_score_fn(
            dim=self.dim,
            score_fn_type=self.score_fn_type,
            dropout=self.dropout,
        )

        # Projection to simplex
        self.simplex_proj_fn = simplex_proj_fn(
            beta=self.beta,
        )
