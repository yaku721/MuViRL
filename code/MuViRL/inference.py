"""Representation inference with an AV-HuBERT-compatible checkpoint."""
import argparse
from importlib import import_module
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def load_model(checkpoint, label_dir, device, fairseq_source=None):
    if fairseq_source:
        sys.path.insert(0, str(Path(fairseq_source).resolve()))
    from fairseq import checkpoint_utils, utils
    from omegaconf import open_dict
    root = Path(__file__).resolve().parent
    utils.import_user_module(SimpleNamespace(user_dir=str(root / 'avhubert')))
    overrides = {'data': str(label_dir), 'label_dir': str(label_dir)}
    state = checkpoint_utils.load_checkpoint_to_cpu(str(checkpoint), overrides)
    tasks = [state['cfg'].task]
    nested = state['cfg'].model.get('w2v_args', None)
    if nested is not None:
        tasks.append(nested.task)
        with open_dict(state['cfg'].model):
            state['cfg'].model.encoder_embed_dim = nested.model.encoder_embed_dim
    for task in tasks:
        with open_dict(task):
            task.pop('input_modality', None)
            task.label_dir = str(label_dir)
            task.data = str(label_dir)
    models, _, task = checkpoint_utils.load_model_ensemble_and_task(
        [str(checkpoint)], arg_overrides=overrides, state=state)
    return models[0].to(device).eval(), task


def load_adapter(wrapper, path):
    state = torch.load(path, map_location='cpu')
    values = state.get('state_dict', state)
    current = wrapper.state_dict()
    unknown = set(values) - set(current)
    if unknown or not values:
        raise ValueError(f'Incompatible adapter keys: {sorted(unknown)[:8]}')
    for name, value in values.items():
        if value.shape != current[name].shape:
            raise ValueError(f'Incompatible adapter shape: {name}')
    wrapper.load_state_dict(values, strict=False)
    return len(values)


@torch.inference_mode()
def run(args):
    torch.set_num_threads(args.threads)
    model, task = load_model(args.checkpoint, args.label_dir, args.device, args.fairseq_source)
    from avhubert.hubert_mv import AVHubertMV
    utils = import_module('avhubert.utils')
    backbone = model.encoder.w2v_model if hasattr(model, 'decoder') else model
    wrapper = AVHubertMV(backbone).to(args.device).eval()
    loaded = load_adapter(wrapper, args.adapter) if args.adapter else 0
    transform = utils.Compose([
        utils.Normalize(0., 255.),
        utils.CenterCrop((task.cfg.image_crop_size,) * 2),
        utils.Normalize(task.cfg.image_mean, task.cfg.image_std)])
    if args.video.suffix.lower() == '.npz':
        with np.load(args.video, allow_pickle=False) as data:
            pixels = data['frames']
        if pixels.ndim != 3 or pixels.dtype != np.uint8:
            raise ValueError('NPZ frames must be uint8 grayscale T,H,W lip crops')
    else:
        pixels = utils.load_video(str(args.video))
    if len(pixels) == 0:
        raise ValueError('The input video contains no decoded frames')
    video = torch.from_numpy(transform(pixels)).float()[None, None].to(args.device)
    frontend = backbone.forward_features(video, modality='video').transpose(1, 2)
    padding = torch.zeros(frontend.shape[:2], device=args.device, dtype=torch.bool)
    encoded = wrapper.encode(frontend, padding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, frontend=frontend[0].cpu().numpy(),
                        representation=encoded[0].cpu().numpy())
    metadata = dict(video=args.video.name, frames=int(video.shape[2]),
                    frontend_shape=list(frontend.shape[1:]),
                    representation_shape=list(encoded.shape[1:]),
                    checkpoint=args.checkpoint.name, adapter=args.adapter.name if args.adapter else None,
                    adapter_tensors_loaded=loaded, audio_input=False,
                    input='Aligned lip crop video; original frame order is preserved')
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True, help='Aligned lip video or NPZ with frames[T,H,W]')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--adapter', type=Path, help='Compatible MuViRL student adapter')
    p.add_argument('--label-dir', type=Path, default=Path(__file__).parent / 'assets/dictionaries')
    p.add_argument('--fairseq-source', type=Path, help='Installed AV-HuBERT-compatible fairseq source')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
