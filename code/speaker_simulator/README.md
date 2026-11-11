# speaker_simulator


## 1. Scope

This compact package renders multiple synchronized views from an existing fitted
Gaussian avatar. It includes the renderer model, dynamic U-Net, camera transforms,
and MP4 export. The avatar contains the fixed pose/expression trajectory and fitted
appearance parameters. 

## 2. Dependencies and assets

Use Linux, Python 3.10, PyTorch 2.1.2, CUDA 12.1, and FFmpeg/FFprobe. Install
`requirements.txt` and compatible PyTorch3D, `simple_knn`, and the **modified**
`diff_gaussian_rasterization` extension distributed by STAvatar. The upstream
installation instructions and extension sources are available at
https://github.com/JiankuoZhao/STAvatar.

After installing the requirements, prepare the compatibility copy used to load
legacy FLAME pickle files:

```bash
python prepare_chumpy.py
```

The script writes `.deps/chumpy` inside this package. The rendering entry point
loads it automatically and does not modify the environment's installed copy.

The FLAME implementation and model assets remain outside this package. Obtain them
under their applicable license. `--flame-models-root` identifies a directory with
`flame/flame.py`, its `lbs.py` dependency, and `flame/weights/`. The implementation
must provide the `FlameHead` and `ASSETS` interfaces used by the STAvatar backend.
Required assets include the FLAME model, UV masks, landmark embeddings, and template
mesh. Keep the corresponding asset license with that installation.

## 3. Avatar input

The saved avatar directory has this layout:

```text
avatar/
  cfg_args
  source_cameras.json
  point_cloud/iteration_N/
    point_cloud.ply
    flame_param.npz
  param/iteration_N/
    dual_branch.pth
    uv_coords.pt
    render_state.json
```



## 4. Render

```bash
CUDA_VISIBLE_DEVICES=0 python render.py \
  --model-path /data/avatar \
  --source-path /data/exported_sequence \
  --flame-models-root /opt/licensed_models \
  --native-libs /opt/stavatar_extensions \
  --source-id trainval/example/50001 \
  --yaws -20 -15 -10 -5 0 5 10 15 20 \
  --pitch 0 \
  --output-path /data/new_rendering
```


