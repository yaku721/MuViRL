import torch
from torch import nn
from torch.nn import functional as F

from .view_aggregation_mv import SetViewAggregation
from .masking_mv import replace_masked_frames


class AVHubertMV(nn.Module):
    def __init__(self, backbone, hidden=128, mask_mode='replace'):
        super().__init__()
        self.backbone = backbone
        self.mask_mode = mask_mode
        self.aggregation = SetViewAggregation(backbone.encoder_embed_dim, hidden)
        initial = backbone.mask_emb.detach() if backbone.mask_emb.numel() == backbone.encoder_embed_dim else torch.zeros(backbone.encoder_embed_dim)
        self.feature_mask = nn.Parameter(initial.clone(), requires_grad=mask_mode == 'token')


    def extract_views(self, videos):
        batch, views, channels, steps, height, width = videos.shape
        x = videos.reshape(batch * views, channels, steps, height, width)
        x = self.backbone.forward_features(x, modality='video')
        return x.transpose(1, 2).reshape(batch, views, steps, -1)


    def encode(self, visual, padding, audio=None, mask=None, generator=None):
        if audio is None:
            audio = torch.zeros_like(visual)
        if self.backbone.modality_fuse == 'concat':
            features = torch.cat((audio, visual), dim=-1)
        else:
            features = audio + visual
        x = self.backbone.layer_norm(features)
        if self.backbone.post_extract_proj is not None:
            x = self.backbone.post_extract_proj(x)
        x = self.backbone.dropout_input(x)
        if mask is not None:
            if self.mask_mode == 'replace':
                x = replace_masked_frames(x, mask, padding, self.training, generator)
            else:
                x = torch.where(mask.unsqueeze(-1), self.feature_mask.to(x.dtype), x)
        x = x.masked_fill(padding.unsqueeze(-1), 0)
        return self.backbone.encoder(x, padding_mask=padding)[0]


    def logits(self, x):
        labels = self.backbone.label_embs_concat.split(self.backbone.num_classes, 0)[0]
        projected = self.backbone.final_proj(x)
        if self.backbone.untie_final_proj:
            projected = projected.chunk(len(self.backbone.num_classes), dim=-1)[0]
        if self.backbone.sim_type == 'cosine':
            projected = F.normalize(projected.float(), dim=-1)
            labels = F.normalize(labels.float(), dim=-1)
        return (projected @ labels.T) / self.backbone.logit_temp


