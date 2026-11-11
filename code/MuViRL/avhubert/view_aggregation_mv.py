import torch
from torch import nn
from torch.nn import functional as F


class SetViewAggregation(nn.Module):
    def __init__(self, dimension, hidden=128):
        super().__init__()
        self.temporal = nn.Conv1d(dimension, hidden, 3, padding=1)
        self.linear1 = nn.Linear(hidden, hidden)
        self.linear2 = nn.Linear(hidden, 1)
        self.context = nn.Linear(hidden * 2, hidden)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, features, valid=None, mode='view_agreement', reliability=None,
                padding=None, prior_strength=1.0):
        batch, views, steps, dim = features.shape
        if valid is None:
            valid = torch.ones(batch, views, steps, device=features.device, dtype=torch.bool)
        if valid.ndim == 2:
            valid = valid.unsqueeze(-1).expand(-1, -1, steps)
        valid = valid.bool()
        if padding is None:
            padding = torch.zeros(batch, steps, device=features.device, dtype=torch.bool)
        keep = valid & ~padding[:, None]
        if ((~keep.any(1)) & ~padding).any():
            raise ValueError('Every nonpadding frame needs an available view')
        clean = features.masked_fill(~keep.unsqueeze(-1), 0)
        if mode == 'view_agreement':
            x = clean.reshape(batch * views, steps, dim).transpose(1, 2)
            x = self.temporal(x).transpose(1, 2).reshape(batch, views, steps, -1)
            context = (x * keep.unsqueeze(-1)).sum(1, keepdim=True)
            context = context / keep.sum(1, keepdim=True).unsqueeze(-1).clamp_min(1)
            x = F.relu(self.context(torch.cat((x, x - context), dim=-1)))
            logits = F.logsigmoid(self.linear2(F.relu(self.linear1(x))).squeeze(-1))
        elif mode == 'uniform':
            logits = features.new_zeros(batch, views, steps)
        else:
            raise ValueError(mode)
        if reliability is not None:
            if reliability.ndim == 2:
                reliability = reliability.unsqueeze(-1).expand(-1, -1, steps)
            if reliability.shape != logits.shape or not torch.isfinite(reliability).all() or (reliability < 0).any():
                raise ValueError('Reliability must be finite and nonnegative with shape B,V or B,V,T')
            weighted = reliability * keep
            reliability = torch.where(weighted.sum(1, keepdim=True) > 0, reliability,
                                      torch.ones_like(reliability))
            logits = logits + prior_strength * reliability.clamp_min(1e-8).log()
        logits = logits.masked_fill(~keep, -torch.finfo(logits.dtype).max)
        weights = logits.softmax(1) * keep
        return (weights.unsqueeze(-1) * clean).sum(dim=1), weights

