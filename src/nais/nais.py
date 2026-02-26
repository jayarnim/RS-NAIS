from typing import Literal
import torch
from components.histories import Histories
from components.base import BaseModel
from .layers.embedding import IDXEmbeddingWithHistory
from .layers.att import AttentionMechanism
from .layers.matching import build as build_matching_layer
from .layers.prediction import ProjectionLayer


class NeuralAttentiveItemSimilarityModels(BaseModel):
    def __init__(
        self,
        histories: Histories,
        num_items: int,
        embedding_dim: int,
        score: Literal["prod", "concat"],
        beta: float,
        dropout: float,
    ):
        """
        NAIS: Neural attentive item similarity model for recommendation (He et al., 2018)
        -----
        Implements the base structure of Neural Attentive Item Similarity Model (NAIS),
        MF & id embedding based collaborative filtering model,
        applying attention mechanism to aggregate histories.

        Args:
            num_items (int):
                total number of items in the dataset, I.
            embedding_dim (int):
                dimensionality of user and item latent representation vectors.
            score (str):
                attention score function type, `prod` or `concat`.
            beta (float):
                smoothing factor for normalization @ simplex.
                (range: (0,1])
            dropout (float):
                dropout rate applied to MLP layers for regularization @ attention score function.
            histories (torch.Tensor): 
                historical item interactions for each user, represented as item indices.
                (shape: [U, history_length])
        """
        super().__init__(locals())

        # HISTORY IDX VIEWER ==========
        self.histories = histories

        # IDX EMBEDDING ==========
        self.embedding = IDXEmbeddingWithHistory(
            num_items=num_items,
            embedding_dim=embedding_dim,
        )

        # HISTORY POOLING ==========
        self.pooling = AttentionMechanism(
            score=score,
            dim=embedding_dim,
            beta=beta,
            dropout=dropout,
        )

        # BILINEAR MATCHING FUNCTION ==========
        self.matching = build_matching_layer(
            name="mf",
        )

        # PREDICTION ==========
        self.prediction = ProjectionLayer(
            dim=embedding_dim,
        )

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        # SEARCH HISTORY IDX ==========
        hist_idx, mask = self.histories(user_idx, item_idx)
        # IDX EMBEDDING ==========
        item_emb, hist_emb = self.embedding(item_idx, hist_idx)
        # HISTORY POOLING ==========
        user_pooled = self.pooling(item_emb, hist_emb, hist_emb, mask)
        # BILINEAR MATCHING FUNCTION ==========
        X_pred = self.matching(user_pooled, item_emb)
        # PRED VEC ==========
        return X_pred

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """   
        # INTERACTION MODELING
        X_pred = self.forward(user_idx, item_idx)
        # PREDICTION
        logit = self.prediction(X_pred)
        return logit
