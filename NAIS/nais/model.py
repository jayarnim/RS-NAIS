import torch
import torch.nn as nn
from ..attn.attn_mechanism import Module as ATTN
from .constants import SCORE_FN_TYPE


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        trn_pos_per_user: torch.Tensor,
        score_fn_type: SCORE_FN_TYPE = "concat",
        beta: float = 0.5,
        dropout: float = 0.2,
    ):
        super(Module, self).__init__()
        
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.trn_pos_per_user = trn_pos_per_user.to(self.device)
        self.score_fn_type = score_fn_type
        self.beta = beta
        self.dropout = dropout

        # generate layers
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        return self._score(user_idx, item_idx)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        with torch.no_grad():
            logit = self._score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def _score(self, user_idx, item_idx):
        # user history: (B, H)
        user_histories = self.trn_pos_per_user[user_idx]
        
        # mask to current target item from history
        mask_target = user_histories == item_idx.unsqueeze(1)
        # mask to padding
        mask_padding = user_histories == self.n_items
        # final mask
        mask = mask_target | mask_padding

        # Embeddings
        p_i = self.embed_target(item_idx)              # (B, D)
        q_j = self.embed_hist(user_histories)          # (B, H, D)

        # Attention
        context = self.attn(p_i, q_j, q_j, mask)

        # Final score
        logit = (
            (context * p_i)
            .sum(dim=1, keepdim=True)        # (B, 1)
            .squeeze(-1)
        )
        
        return logit

    def _init_layers(self):
        # Item embeddings
        kwargs = dict(
            num_embeddings=self.n_items+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_items,
        )
        self.embed_target = nn.Embedding(**kwargs)
        self.embed_hist = nn.Embedding(**kwargs)

        # attn
        kwargs = dict(
            dim=self.n_factors,
            score_fn_type=self.score_fn_type,
            beta=self.beta,
            dropout=self.dropout,
        )
        self.attn = ATTN(**kwargs)