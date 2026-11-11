"""Score paired synchronized lip videos with a locally installed Laya Vision model."""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from aggregate import utility_and_reliability


def frames(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open {path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    output = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        output.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not output or fps <= 0:
        raise ValueError('Empty video or invalid frame rate')
    return output, fps


def grid(images, indices, cell=160):
    canvas = Image.new('RGB', (cell * 3, cell * 3), 'white')
    for j, index in enumerate(indices):
        patch = ImageOps.contain(images[index], (cell, cell), Image.Resampling.LANCZOS)
        x = (j % 3) * cell + (cell - patch.width) // 2
        y = cell // 2 + (j // 3) * cell + (cell - patch.height) // 2
        canvas.paste(patch, (x, y))
    return canvas


@torch.inference_mode()
def run(args):
    if args.segments < 1:
        raise ValueError('segments must be positive')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if args.laya_source:
        sys.path.insert(0, str(args.laya_source.resolve()))
    import laya
    reference, fps = frames(args.original)
    candidate, candidate_fps = frames(args.candidate)
    if len(reference) != len(candidate) or abs(fps - candidate_fps) > 1e-3:
        raise ValueError('The paired videos must preserve frame count and frame rate')
    agent = laya.load_vlm(str(args.checkpoint.resolve()), device=args.device,
        preprocess='gpu', image_interpolation='processor', image_size=512, image_split_edge=0)
    if args.adapter:
        adapter = torch.load(args.adapter, map_location='cpu')
        if adapter.get('settings', {}).get('layout') != 'six_frames_2x3_square_v1':
            raise ValueError('Decision adapter uses another temporal layout')
        current = agent.model.state_dict()
        state = adapter['state_dict']
        if not state or set(state) - set(current):
            raise ValueError('Decision adapter keys do not match the checkpoint')
        agent.model.load_state_dict(state, strict=False)
    agent.model.eval()
    prompt = json.loads((Path(__file__).parent / 'decision_prompt.json').read_text())
    bounds = np.linspace(0, len(reference), min(args.segments, max(1, len(reference) // 6)) + 1).astype(int)
    scores = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        indices = np.linspace(start, end - 1, min(6, end-start)).round().astype(int).tolist()
        state = dict(images=[grid(reference, indices), grid(candidate, indices)],
            note=f'First grid: reference. Second grid: candidate. Both use synchronized frame IDs {indices}. '
                 f'Six cells run left to right, then top to bottom. Camera yaw {args.yaw:g}, pitch {args.pitch:g} degrees.')
        raw = {}
        agent.predict(state, {'articulation': prompt}, strict=True, _raw_logits=raw)
        logits = torch.as_tensor(raw['articulation']).float()
        utility, reliability = utility_and_reliability(logits, args.temperature,
            torch.tensor(args.geometry, device=logits.device), args.minimum)
        scores.append(dict(start_frame=int(start), end_frame=int(end), frame_ids=indices,
            logits=logits.cpu().tolist(), utility=float(utility), reliability=float(reliability)))
    result = dict(original=args.original.name, candidate=args.candidate.name, fps=fps,
        frames=len(reference), temperature=args.temperature, geometry=args.geometry,
        minimum=args.minimum, prompt='decision_prompt.json', segments=scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--laya-source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--segments', type=int, default=3)
    p.add_argument('--yaw', type=float, default=0.)
    p.add_argument('--pitch', type=float, default=0.)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--geometry', type=float, default=1.)
    p.add_argument('--minimum', type=float, default=.25)
    p.add_argument('--device', default='cuda')
    run(p.parse_args())
