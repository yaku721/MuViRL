"""Run Calibrated View Aggregation on synchronized frontend features."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch


def aggregation_class():
    # The isolated import keeps cached-feature inference independent of fairseq.
    path = Path(__file__).parent / 'avhubert/view_aggregation_mv.py'
    spec = importlib.util.spec_from_file_location('muvirl_aggregation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SetViewAggregation


def utility_and_reliability(logits, temperature=1., geometry=None, minimum=.25):
    if logits.shape[-1] != 4 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite logits with final dimension four')
    if not np.isfinite(temperature) or temperature <= 0 or not 0 < minimum < 1:
        raise ValueError('Temperature must be positive and 0 < minimum < 1')
    probabilities = (logits.float() / temperature).softmax(-1)
    levels = torch.arange(4, device=logits.device, dtype=probabilities.dtype) / 3
    utility = probabilities @ levels
    geometry = torch.ones_like(utility) if geometry is None else geometry
    if geometry.shape != utility.shape or not torch.isfinite(geometry).all() or ((geometry < 0) | (geometry > 1)).any():
        raise ValueError('Geometry must match utility shape and lie in [0, 1]')
    return utility, minimum + (1 - minimum) * utility * geometry


@torch.inference_mode()
def run(args):
    data = np.load(args.input, allow_pickle=False)
    features = torch.from_numpy(data['features']).float()
    if features.ndim != 4 or not torch.isfinite(features).all():
        raise ValueError('features must be finite with shape B,V,T,C')
    batch, views, steps, dim = features.shape
    valid = torch.from_numpy(data['valid']).bool() if 'valid' in data else None
    padding = torch.from_numpy(data['padding']).bool() if 'padding' in data else None
    reliability = torch.from_numpy(data['reliability']).float() if 'reliability' in data else None
    if 'logits' in data:
        if reliability is not None:
            raise ValueError('Supply logits or reliability, not both')
        logits = torch.from_numpy(data['logits']).float()
        geometry = torch.from_numpy(data['geometry']).float() if 'geometry' in data else None
        _, reliability = utility_and_reliability(logits, args.temperature, geometry, args.minimum)
    if args.original_view is not None:
        if not 0 <= args.original_view < views:
            raise ValueError('original-view must identify an input view')
        if reliability is None:
            reliability = torch.ones(batch, views, steps)
        reliability[:, args.original_view] = 1.
    model = aggregation_class()(dim, args.hidden).eval()
    mode = 'uniform'
    if args.mode == 'adaptive':
        if args.weights is None:
            raise ValueError('Adaptive Weighting requires trained --weights')
        state = torch.load(args.weights, map_location='cpu')
        values = state.get('state_dict', state)
        if any(k.startswith('aggregation.') for k in values):
            values = {k[len('aggregation.'):]: v for k, v in values.items() if k.startswith('aggregation.')}
        required = {k for k in model.state_dict() if k.startswith(('temporal.', 'linear1.', 'linear2.', 'context.'))}
        if required - set(values) or set(values) - set(model.state_dict()):
            raise ValueError('Incomplete or incompatible Adaptive Weighting state')
        model.load_state_dict(values, strict=False)
        mode = 'view_agreement'  # Checkpoint-compatible key for Adaptive Weighting.
    output, weights = model(features, valid, mode=mode, reliability=reliability,
                            padding=padding, prior_strength=.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, aggregate=output.numpy(), weights=weights.numpy())
    print(f'Aggregate {tuple(output.shape)}; weights {tuple(weights.shape)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['mean', 'adaptive'], default='adaptive')
    p.add_argument('--weights', type=Path)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--minimum', type=float, default=.25)
    p.add_argument('--original-view', type=int, help='Index of the original view, whose reliability is fixed to one')
    run(p.parse_args())
