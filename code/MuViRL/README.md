# MuViRL

Run the commands below from this directory.

## 1. Environment

Use Python 3.10 and a PyTorch environment. Install `requirements.txt`, then install
the fairseq version supplied with AV-HuBERT, or pass its source directory through
`--fairseq-source`. The selected model definitions use the fairseq 0.12.2 interface.
The original sources are available at https://github.com/facebookresearch/av_hubert.

`aggregate.py` and `smoke_test.py` need only NumPy and PyTorch. Laya decision inference
uses a separate environment described by `requirements_decision.txt`, a local Laya
Vision installation, and a checkpoint. Follow its official
instructions at https://github.com/r33drichards/laya-vision.

## 2. Extract representations from one video

Use aligned lip crops, retaining the original frame order. Inputs may be a video or
an NPZ file containing `frames`, a `uint8` array of shape `T,H,W`. The model's image
normalization and central crop are read from its checkpoint configuration.

```bash
python inference.py \
  --video /data/example/lips.mp4 \
  --checkpoint /models/avhubert/base_lrs3_iter5.pt \
  --adapter ../../models/adapter.pt \
  --fairseq-source /opt/av_hubert/fairseq \
  --device cuda \
  --output /data/output/representation.npz
```


## 3. Aggregate cached view features

An NPZ input contains synchronized `features[B,V,T,C]`. Optional arrays are
`valid[B,V]` or `valid[B,V,T]`, `padding[B,T]`, and
`reliability[B,V]` or `reliability[B,V,T]`. Here `True` means valid in `valid` and
padding in `padding`. At least one valid view must remain at every unpadded frame.


```bash
python aggregate.py \
  --input /data/example/frontend_features.npz \
  --weights ../../models/adapter.pt \
  --mode adaptive \
  --original-view 0 \
  --output /data/output/aggregation.npz
```

`adaptive` uses trained Adaptive Weighting parameters. The implementation key
`view_agreement` is retained for checkpoint compatibility. Reliability enters as
its square root, matching the manuscript's aggregation equation. The output
contains `aggregate[B,T,C]` and normalized `weights[B,V,T]`.


## 4. Score synchronized lip videos

```bash
python decision.py \
  --original /data/example/original_lips.mp4 \
  --candidate /data/example/synthesized_lips.mp4 \
  --checkpoint /models/laya_vision \
  --adapter /models/muvirl/decision_adapter.pt \
  --laya-source /opt/laya-vision \
  --temperature 1.0 \
  --yaw 10 \
  --output /data/output/decision.json
```


