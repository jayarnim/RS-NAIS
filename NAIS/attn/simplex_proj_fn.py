import torch
import torch.nn as nn



class Module(nn.Module):
    def __init__(
        self,
        beta: float=0.5,
    ):
        super().__init__()
        self.beta = beta

    def forward(self, scores):
        numerator = torch.exp(scores)
        numerator_sum = numerator.sum(dim=-1, keepdim=True)
        denominator = numerator_sum ** self.beta
        return numerator / denominator