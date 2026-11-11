import argparse
import copy
import json
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from tqdm import tqdm

from gaussian_renderer import render
from local.avatar_backend import load_checkpoint
from local.prepare_view_transforms import rotate_camera
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import cameraList_from_camInfos


def load_views(args, model_args):
    if args.views_list:
        names = args.views_list.read_text().splitlines()
        folders = [args.views_list.parent/name for name in names]
        metadata = [json.loads((folder/'view.json').read_text()) for folder in folders]
        source = args.source_path or Path(model_args.source_path)
        if any(Path(row['source']).resolve() != source.resolve() for row in metadata):
            raise ValueError('Views must describe this source sequence')
        infos = []
        for folder in folders:
            group = []
            for split in ('train', 'val', 'test'):
                file = f'transforms_{split}.json'
                if (folder/file).is_file():
                    group.extend(readCamerasFromTransforms(str(folder), file, model_args.white_background))
            infos.append(group)
    else:
        saved = args.model_path/'source_cameras.json'
        source = json.loads(saved.read_text())
        original_root = Path(source['source_path'])
        image_root = (args.source_path or original_root).resolve()
        fps = args.fps or source.get('fps')
        if not fps or fps <= 0:
            raise ValueError('Specify --fps when the model has no saved frame rate')
        source_id = args.source_id
        if not source_id:
            video = Path(source['source_video'])
            parts = [video.parent.name, video.stem]
            if video.parent.parent.name in ('trainval', 'test', 'pretrain', 'train', 'val'):
                parts.insert(0, video.parent.parent.name)
            source_id = '/'.join(parts)
        if Path(source_id).is_absolute() or '..' in Path(source_id).parts:
            raise ValueError('source-id must be a relative video name')
        clip_id = args.clip_id or args.model_path.name
        names, metadata, infos = [], [], []
        folders = [saved]*len(args.yaws)
        with tempfile.TemporaryDirectory(prefix='speaker_cameras_') as temporary:
            for yaw in args.yaws:
                name = f'{clip_id}_yaw_{yaw:g}_pitch_{args.pitch:g}'
                names.append(name)
                data = copy.deepcopy(source)
                for frame in data['frames']:
                    image_path = Path(frame['file_path'])
                    if image_path.is_absolute():
                        image_path = image_path.relative_to(original_root)
                    frame['file_path'] = str(image_root / image_path)
                    frame['transform_matrix'] = rotate_camera(frame['transform_matrix'], yaw, args.pitch, np.array(args.pivot))
                (Path(temporary)/'transforms_train.json').write_text(json.dumps(data))
                infos.append(readCamerasFromTransforms(temporary, 'transforms_train.json', model_args.white_background))
                metadata.append(dict(clip_id=clip_id, source_id=source_id,
                    video_file=f'{source_id}_a{yaw:g}_e{args.pitch:g}.mp4', expected_yaws=args.yaws,
                    source=str(image_root), fps=fps, yaw=yaw, pitch=args.pitch, pivot=args.pivot,
                    transform='world-space camera orbit: Ry(yaw)Rx(pitch)',
                    flame_motion='checkpoint', frames=len(data['frames'])))
    if not names or len(names) != len(set(names)):
        raise ValueError('Camera views must be nonempty and unique')
    cameras = [cameraList_from_camInfos(group, 1.0, model_args) for group in infos]
    return names, folders, metadata, cameras


@torch.no_grad()
def run(args):
    started = time.perf_counter()
    gs, network, model_args, iteration = load_checkpoint(args.model_path)
    names, folders, metadata, cameras = load_views(args, model_args)
    order = [c.timestep for c in cameras[0]]
    if not order or any(row.get('frames', len(order)) != len(order) for row in metadata):
        raise ValueError('View metadata and camera frame counts differ')
    if any([c.timestep for c in group] != order for group in cameras[1:]):
        raise ValueError('View frame orders differ')
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.tensor([1,1,1] if model_args.white_background else [0,0,0], dtype=torch.float32, device='cuda')
    args.output_path.mkdir(parents=True, exist_ok=True)
    (args.output_path/'metadata').mkdir(exist_ok=True)
    videos = [args.output_path/row['video_file'] for row in metadata]
    temporary = [p.with_name(p.stem+'.partial.mp4') for p in videos]
    for name, path in zip(names,videos):
        if path.exists() or (args.output_path/'metadata'/f'{name}.json').exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    encoders = []
    torch.cuda.reset_peak_memory_stats()
    shared_seconds = 0.
    raster_seconds = [0.]*len(names)
    frame_seconds = [0.]*len(names)
    frames = [[] for _ in names]
    try:
        for index, (group, row, path) in enumerate(zip(cameras,metadata,temporary)):
            camera = group[0]
            encoders.append(subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24',
                '-video_size',f'{camera.image_width}x{camera.image_height}','-framerate',str(row['fps']),
                '-i','pipe:0','-an','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p',
                '-threads','2','-movflags','+faststart',str(path)],stdin=subprocess.PIPE))
        for index, timestep in enumerate(tqdm(order,desc=f'Rendering {len(names)} views together')):
            torch.cuda.synchronize(); step = time.perf_counter()
            gs.select_mesh_by_timestep(timestep)
            offset = network._decode(gs.flame_param,timestep,gs.get_vertex_displace_map())
            torch.cuda.synchronize(); shared_seconds += time.perf_counter()-step
            for view_index, group in enumerate(cameras):
                camera = group[index]
                step = time.perf_counter()
                output = render(camera,gs,pipe,background,offset=offset)['render'].clamp(0,1)
                torch.cuda.synchronize(); raster_seconds[view_index] += time.perf_counter()-step
                pixels = output.mul(255).add(.5).clamp(0,255).byte().permute(1,2,0).cpu().numpy()
                encoders[view_index].stdin.write(pixels.tobytes())
                frame_seconds[view_index] += time.perf_counter()-step
                frames[view_index].append(dict(index=index,timestep=int(timestep),source_image=camera.image_path))
        for encoder in encoders:
            encoder.stdin.close()
        failures = [encoder.wait() for encoder in encoders]
        if any(failures):
            raise RuntimeError(f'Video encoding failed: {failures}')
        for source,target in zip(temporary,videos):
            source.replace(target)
        encoders=[]
    finally:
        for encoder in encoders:
            if encoder.poll() is None:
                encoder.kill(); encoder.wait()
        for path in temporary:
            path.unlink(missing_ok=True)
    wall = time.perf_counter()-started
    for index,name in enumerate(names):
        seconds = raster_seconds[index]+shared_seconds/len(names)
        report = dict(video_file=metadata[index]['video_file'],frames=frames[index],fps=metadata[index]['fps'],
            iteration=iteration,model_path=str(args.model_path),target_path=str(folders[index]),
            active_sh_degree=gs.active_sh_degree,sh_degree_source='checkpoint state or legacy schedule',
            motion_source='checkpoint',reference_kind='source_camera_reference',save_intermediates=False,
            render_seconds=seconds,render_fps=len(order)/seconds,attributed_frame_seconds=frame_seconds[index]+shared_seconds/len(names),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
            view=metadata[index],renderer='shared_model_shared_frame',group_wall_seconds=wall,
            time_scope='shared decode and view raster/encoding writes; group wall includes loading and encoder drain')
        report_file = args.output_path/'metadata'/f'{name}.json'
        temporary_report = report_file.with_suffix('.partial')
        temporary_report.write_text(json.dumps(report,indent=2)+'\n')
        temporary_report.replace(report_file)
    profile = dict(views=len(names),frames_per_view=len(order),wall_seconds=wall,
        shared_decode_seconds=shared_seconds,raster_seconds=sum(raster_seconds),
        render_fps=len(order)*len(names)/(shared_seconds+sum(raster_seconds)),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
    clip_id = metadata[0].get('clip_id') or args.model_path.name
    profile['clip_id'] = clip_id
    profiles = args.output_path/'render_profiles'
    profiles.mkdir(exist_ok=True)
    temporary_profile = profiles/f'{clip_id}.partial'
    temporary_profile.write_text(json.dumps(profile,indent=2)+'\n')
    temporary_profile.replace(profiles/f'{clip_id}.json')
    from local.layout import mark_completed
    mark_completed(args.output_path, clip_id)
    print(profile,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-path',type=Path,required=True)
    parser.add_argument('--source-path',type=Path)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--views-list',type=Path)
    group.add_argument('--yaws',type=float,nargs='+')
    parser.add_argument('--pitch',type=float,default=0)
    parser.add_argument('--pivot',type=float,nargs=3,default=[0,0,0])
    parser.add_argument('--fps',type=float)
    parser.add_argument('--source-id')
    parser.add_argument('--clip-id')
    parser.add_argument('--output-path',type=Path,required=True)
    args=parser.parse_args()
    from local.run_stats import start_profile
    start_profile()
    run(args)
