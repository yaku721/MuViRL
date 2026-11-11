# MuViRL

**Multimodal View-Invariant Visual Speech Representation Learning via Calibrated View Aggregation**

MuViRL learns visual speech representations across views through multimodal self-supervision. It uses synchronized 3D Gaussian view synthesis, calibrated view aggregation, Aggregation-to-Single Distillation, and real–synthetic alignment.

[Watch introduction video](https://yaku721.github.io/MuViRL/) · [Download MP4 (1080p)](introduction_video/MuViRL_introduction_video_1080p.mp4)

## Contents

| Directory | Contents |
|---|---|
| [code/MuViRL](code/MuViRL/) | Representation extraction, decision scoring, and view aggregation. |
| [code/speaker_simulator](code/speaker_simulator/) | Multiview rendering from fitted 3D Gaussian avatars. |
| [models](models/) | Base encoder adapter and configuration. |
| [results](results/) | Experiment data, verification scripts, and plotting tools. |
| [synthesis_data_examples](synthesis_data_examples/) | Original and synthesized LRS3 video examples. |

## Usage

Recompute the result summaries from the repository root:

```bash
python results/verify_results.py
```

For setup and inference commands, see the [MuViRL guide](code/MuViRL/README.md) and the [rendering guide](code/speaker_simulator/README.md).
