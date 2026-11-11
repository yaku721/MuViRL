import torch


def replace_masked_frames(features, mask, padding, training=False, generator=None):
    replacement = features.clone()
    for index in range(len(features)):
        targets = (mask[index] & ~padding[index]).nonzero(as_tuple=True)[0]
        donors = (~mask[index] & ~padding[index]).nonzero(as_tuple=True)[0]
        if not len(targets):
            continue
        if not len(donors):
            replacement[index, targets] = 0
            continue
        offsets = (torch.randint(len(donors), (len(targets),), device=features.device, generator=generator)
                   if training else torch.arange(len(targets), device=features.device) % len(donors))
        replacement[index, targets] = features[index, donors[offsets]]
    return replacement
