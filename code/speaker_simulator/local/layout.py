from pathlib import Path


def experiment_paths(root, clip_id=None):
    root = Path(root)
    extracted = 'stage1_extract_data'
    if clip_id:
        return dict(clip=root/extracted/clip_id, matte=root/extracted/clip_id,
            model=root/'stage3_training_model'/clip_id, renders=root/'stage4_simulated_data',
            views=root/'stage4_simulated_data/cameras', tracking=root/'stage2_tracking_data'/clip_id,
            dataset=root/'stage2_tracking_data/working_data'/clip_id, quality=root/'stage5_check_data')
    if (root / 'stage1_extract_data').exists() or (root / 'stage0_filtered_data').exists():
        names = dict(clip='stage1_extract_data/clip', matte='stage1_extract_data/clip',
                     model='stage3_training_model', renders='stage4_simulated_data',
                     views='stage4_simulated_data/cameras', tracking='stage2_tracking_data/tracking',
                     dataset='stage2_tracking_data/dataset', quality='stage5_check_data')
    elif (root / 'stage1_frames').exists():
        names = dict(clip='stage1_frames/clip', matte='stage2_matte/clip', model='stage5_model',
                     renders='stage6_simulated_data', views='stage6_simulated_data/cameras',
                     tracking='stage3_tracking', dataset='stage4_dataset', quality='stage7_quality')
    else:
        names = dict(clip='data/clip', matte='data/clip', model='model', renders='simulated_data',
                     views='views', tracking='tracking', dataset='export', quality='quality')
    return {key: root / value for key, value in names.items()}


def render_metadata(root, name):
    import json
    path = Path(root) / 'metadata' / f'{name}.json'
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        pass
    clip_id = name.rsplit('_yaw_', 1)[0]
    data = json.loads((Path(root) / 'metadata' / f'{clip_id}.json').read_text())
    return dict(data['views'][name], frames=data['frames'])


def render_entries(root, clip_id):
    import json
    root = Path(root)
    merged = root / 'metadata' / f'{clip_id}.json'
    entries = json.loads(merged.read_text())['views'] if merged.exists() else {}
    for path in (root / 'metadata').glob(f'{clip_id}_yaw_*.json'):
        try:
            entries[path.stem] = json.loads(path.read_text())
        except FileNotFoundError:
            pass
    if merged.exists():
        entries.update(json.loads(merged.read_text())['views'])
    return entries


def mark_completed(root, clip_id):
    from local.records import record_lock
    root = Path(root)
    entries = render_entries(root, clip_id)
    first = next(iter(entries.values()), {}).get('view', {})
    expected = first.get('expected_views', [[yaw, first.get('pitch', 0)] for yaw in first.get('expected_yaws', [])])
    actual = {(row['view']['yaw'], row['view']['pitch']) for row in entries.values()
              if (root / row['video_file']).is_file()}
    if not first.get('source_id') or not expected or not set(map(tuple, expected)).issubset(actual):
        return
    done = root / 's4done.list'
    with record_lock(done):
        completed = set(done.read_text().splitlines()) if done.exists() else set()
        completed.add(first['source_id'])
        temporary = done.with_suffix('.partial')
        temporary.write_text(''.join(name + '\n' for name in sorted(completed)))
        temporary.replace(done)
