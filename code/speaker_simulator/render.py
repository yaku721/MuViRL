"""Render synchronized views from a fitted speaker_simulator avatar."""
import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--source-path', type=Path, required=True,
                   help='Relocated exported image dataset used by this avatar')
    p.add_argument('--flame-models-root', type=Path, required=True,
                   help='Licensed package root containing flame/flame.py and flame/weights/')
    p.add_argument('--native-libs', type=Path, help='Installed CUDA extension directory')
    p.add_argument('--output-path', type=Path, required=True)
    p.add_argument('--source-id', required=True, help='Relative split/speaker/stem')
    p.add_argument('--clip-id')
    p.add_argument('--yaws', type=float, nargs='+', default=list(range(-20, 21, 5)))
    p.add_argument('--pitch', type=float, default=0)
    p.add_argument('--pivot', type=float, nargs=3, default=[0, 0, 0])
    p.add_argument('--fps', type=float)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    args.model_path = args.model_path.resolve()
    args.source_path = args.source_path.resolve()
    args.output_path = args.output_path.resolve()
    root = Path(__file__).resolve().parent
    for name in ['cfg_args', 'source_cameras.json', 'point_cloud', 'param']:
        if not (args.model_path / name).exists():
            p.error(f'Missing avatar file or directory: {name}')
    if not (args.flame_models_root / 'flame/flame.py').is_file():
        p.error('--flame-models-root must contain flame/flame.py')
    if not args.source_path.is_dir():
        p.error('The exported source image dataset does not exist')
    if len(set(args.yaws)) != len(args.yaws):
        p.error('Yaw angles must be unique')
    if Path(args.source_id).is_absolute() or '..' in Path(args.source_id).parts:
        p.error('source-id must be a relative split/speaker/stem name')
    if args.output_path.exists() and any(args.output_path.iterdir()):
        p.error('Use an empty output directory')
    if args.validate_only:
        print('Avatar paths and output destination validated')
        return
    # Extend the local model package with the separately licensed FLAME implementation.
    sys.path.insert(0, str(root / 'models/stavatar'))
    if (root / '.deps').is_dir():
        sys.path.insert(0, str(root / '.deps'))
    if args.native_libs:
        sys.path.insert(0, str(args.native_libs.resolve()))
    import models
    models.__path__.append(str(args.flame_models_root.resolve()))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Rendering requires a CUDA GPU and compiled rasterizer')
    args.views_list = None
    from local.render_model_views import run
    run(args)


if __name__ == '__main__':
    main()
