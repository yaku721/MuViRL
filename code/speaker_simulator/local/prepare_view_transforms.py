import argparse
import copy
import json
from pathlib import Path

import numpy as np


def rotate_camera(matrix, yaw, pitch, pivot):
    y, p = np.deg2rad([yaw, pitch])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rotation = ry @ rx
    result = np.array(matrix, dtype=float).copy()
    result[:3, :3] = rotation @ result[:3, :3]
    result[:3, 3] = pivot + rotation @ (result[:3, 3] - pivot)
    return result.tolist()


def run(args):
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames, template = {}, None
    for split in ['train', 'val', 'test']:
        data = json.loads((source / f'transforms_{split}.json').read_text())
        template = data if template is None else template
        for frame in data['frames']:
            key = (frame['timestep_index'], frame['camera_index'])
            if key in frames and frames[key] != frame:
                raise ValueError(f'Conflicting frame {key}')
            frames[key] = frame
    if not frames:
        raise ValueError('No camera frames')
    views = args.views or [[yaw, args.pitch] for yaw in args.yaws]
    if any(len(pair) != 2 for pair in views) or len(set(map(tuple, views))) != len(views):
        raise ValueError('views must contain unique [yaw, pitch] pairs')
    names = []
    for yaw, pitch in views:
        name = (args.clip_id + '_' if args.clip_id else '') + f'yaw_{yaw:g}_pitch_{pitch:g}'
        folder = output / name
        if folder.exists():
            raise FileExistsError(folder)
        folder.mkdir()
        data = copy.deepcopy(template)
        data['frames'] = []
        for key in sorted(frames):
            frame = copy.deepcopy(frames[key])
            frame['transform_matrix'] = rotate_camera(frame['transform_matrix'], yaw, pitch,
                                                       np.array(args.pivot))
            data['frames'].append(frame)
        data['timestep_indices'] = sorted({x['timestep_index'] for x in data['frames']})
        (folder / 'transforms_train.json').write_text(json.dumps(data, indent=2) + '\n')
        empty = dict(data, frames=[], timestep_indices=[])
        for split in ['val', 'test']:
            (folder / f'transforms_{split}.json').write_text(json.dumps(empty, indent=2) + '\n')
        for asset in ['images', 'fg_masks', 'flame_param', 'canonical_flame_param.npz']:
            if (source / asset).exists():
                (folder / asset).symlink_to(source / asset)
        metadata = {'clip_id': args.clip_id, 'source_id': args.source_id,
                    'video_file': (args.source_id + f'_a{yaw:g}_e{pitch:g}.mp4') if args.source_id else name + '.mp4',
                    'expected_views': views, 'source': str(source), 'fps': args.fps, 'yaw': yaw, 'pitch': pitch,
                    'pivot': args.pivot, 'transform': 'world-space camera orbit: Ry(yaw)Rx(pitch)',
                    'flame_motion': 'unchanged, read-only source link', 'frames': len(frames)}
        (folder / 'view.json').write_text(json.dumps(metadata, indent=2) + '\n')
        names.append(name)
        print(folder)
    (output / (f'{args.clip_id}.list' if args.clip_id else 'views.list')).write_text(''.join(name + '\n' for name in names))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-id')
    parser.add_argument('--source-id')
    parser.add_argument('--yaws-override', default='')
    parser.add_argument('--pitch-override', default='')
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fps', type=float)
    parser.add_argument('--video-metadata')
    parser.add_argument('--yaws', type=float, nargs='+', default=[-10, 0, 10])
    parser.add_argument('--pitch', type=float, default=0)
    parser.add_argument('--views', type=json.loads, default=[])
    parser.add_argument('--pivot', type=float, nargs=3, default=[0, 0, 0])
    args = parser.parse_args()
    if args.yaws_override:
        args.yaws = [float(x) for x in args.yaws_override.split()]
        args.views = []
    if args.pitch_override:
        args.pitch = float(args.pitch_override)
        args.views = []
    if args.video_metadata:
        args.fps = json.loads(Path(args.video_metadata).read_text())['fps']
    if args.fps is None or args.fps <= 0:
        parser.error('fps must be positive')
    from local.run_stats import start_profile
    start_profile()
    run(args)
