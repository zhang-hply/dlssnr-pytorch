# dlssnr-pytorch

NVIDIA's DLSS 5 neural rendering network ("DLSS-NR") in PyTorch.

The network takes a colour frame and the library's conditioning controls and returns the enhanced frame, reproducing NVIDIA's output for the same input to about 45 to 49 dB PSNR.

```bash
dlssnr extract ./path/to/nvngx_dlssnr.dll -o weights.safetensors
dlssnr run frame.png -o enhanced.png --side-by-side
dlssnr verify --input frame.png --reference nvidia_output.png
```

[![The input frame on the left and the network's output on the right](docs/steep-comparison-preview.jpg)](docs/steep-comparison.jpg)

A frame from Steep (input on the left, this network's output with the default controls on the right); click for the full-size image. Steep is a trademark and the game's art is the property of Ubisoft; the screenshot is shown only to illustrate the network's output and is not covered by this project's license.

## Results

The same Tomb Raider (2013) frame at three sizes, fed to this network and to NVIDIA's library with the
default controls, compared as 8-bit RGB (single frames, no history). "Unprocessed" is the input against
NVIDIA's output, for scale. The CPU column and the metrics were measured with the reference arithmetic
(`DLSSNR_TRITON=0` gives the same bits on a GPU: 45.55, 48.21 and 49.05 dB, 0.37, 0.70 and 1.82 s on the
7900 XTX); the GPU column is the Triton kernels described under [Speed](#speed), steady state after the first frame.

| Frame | PSNR vs NVIDIA | SSIM vs NVIDIA | Edit correlation | Unprocessed PSNR | 16 CPU threads | RX 7900 XTX (ROCm) |
| --- | --- | --- | --- | --- | --- | --- |
| 1920x1080 | 45.70 dB | 0.9961 | 0.9947 | 26.44 dB | 21 s | 0.19 s |
| 2560x1440 | 48.26 dB | 0.9968 | 0.9956 | 27.72 dB | 40 s | 0.33 s |
| 3840x2160 | 49.10 dB | 0.9970 | 0.9955 | 28.69 dB | 90 s | 0.77 s |

The timings are not a focus of this project and do not represent what a GPU can do with this network. This is a
reference implementation in PyTorch, written to reproduce NVIDIA's output, not to be fast. A dedicated Vulkan port,
[DLSSNR-RDNA3](https://github.com/mauri870/DLSSNR-RDNA3), runs the same network at 4K in 59 ms on the same
7900 XTX.

"Edit correlation" measures how closely this network's changes match NVIDIA's. NVIDIA's output is identical across runs, so the remaining differences come from arithmetic: the network rounds to 8-bit floating point after most operations, and summation order can shift values near rounding boundaries. Against an independent implementation, it matches 97–100% of each layer's output exactly and is within one 8-bit step on nearly all other elements; the widest Swin layers and pre-block have the lowest exact-match rates.

## Setup

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). Pick the PyTorch build for your hardware:

```
uv sync --extra rocm    # Radeon RX 7000/9000, Instinct (ROCm 7.2 wheels)
uv sync --extra cuda    # NVIDIA (CUDA 13.0 wheels)
uv sync --extra cpu     # no GPU
```

`--device auto` (the default) uses the GPU when PyTorch sees one and the CPU otherwise; `cpu`, `cuda` and
`cuda:N` force a choice (ROCm builds also report `cuda`). The GPU results above were measured on a
7900 XTX with ROCm; the CUDA build runs the same code and has not been tried on an NVIDIA card.
GPU memory at 1080p, 1440p and 4K peaks at 1.7, 2.6 and 5.0 GiB (2.6, 4.3 and 8.8 GiB with the reference arithmetic).

## Speed

On an AMD GPU (the ROCm build) the layers that dominate the time run as Triton kernels (`dlssnr/gemm.py`,
`attention.py`, `swin_fused.py`; Triton ships with the ROCm PyTorch wheel). The CUDA build and the CPU use the
reference arithmetic:

- every matrix product reads its e4m3 operands as float16 (exact) on the matrix units with a float32
  accumulator, and the activation and the rounding to e4m3 are done in the kernel's epilogue;
- the window attention of the Swin blocks and the global attention of the vision-transformer blocks are one
  kernel each, with the rounding points and the float16 summation orders of the reference;
- the whole body of the single-head 32-channel Swin blocks (the pre-block, the post-block and the layers at
  full and half resolution) is one kernel per attention window, so its feature map is read and written once.

The float32 sums inside the matrix products are taken in the matrix unit's order instead of the reference's,
so a fraction of a percent of the values in a layer land one e4m3 step away, and the difference accumulates
over the 152 layers. Against NVIDIA's output the three frames score within 0.15 dB of the reference
arithmetic (0.15, 0.05 and 0.05 dB higher), which is rounding noise, not a gain. `DLSSNR_TRITON=0` runs the reference arithmetic
(elementwise PyTorch and float32 matrix products); the CPU always does. The first frame also compiles the
kernels (about 4 s, cached on disk afterwards).

On a 7900 XTX at 1080p, the 8 layers at 32 channels, the pre-block and the post-block went from 180 ms to 32 ms, the vision-transformer
attention from 35 to 8 ms, and the frame from 372 to about 190 ms. The multi-head Swin layers (64 to 256
channels, about 100 ms at 1080p) still run as unfused PyTorch operations plus the Triton matrix products and
attention; a one-kernel-per-window version of them was tried and was not faster.

## int4 mixed network (optional)

`--int4` runs the "int4 mixed" network of [DLSSNR-AMD](https://github.com/mauri870/DLSSNR-RDNA3): the heavy
matrices of the vision-transformer blocks, of the C=512 attention and of the C=128 and C=256 Swin blocks
(the MLP expansion and the QKV) run on an int4 grid, and most Swin blocks take a hard-swish for the cubic SiLU
(its scale folded into the weights that follow). It is a different network, not NVIDIA's picture:

```
dlssnr run frame.png --int4 --int4-tables ../DLSSNR-AMD/linux/data/int4/rdna4
python scripts/int4_check.py        # the three verification frames, against NVIDIA and against DLSSNR-AMD's int4 pictures
```

| Frame | vs NVIDIA | DLSSNR-AMD's int4 vs NVIDIA | vs DLSSNR-AMD's int4 picture | exact network vs NVIDIA |
| --- | --- | --- | --- | --- |
| 1920x1080 | 42.01 dB | 41.92 dB | 46.85 dB | 45.70 dB |
| 2560x1440 | 43.49 dB | 43.36 dB | 47.29 dB | 48.26 dB |
| 3840x2160 | 43.39 dB | 43.62 dB | 48.26 dB | 49.10 dB |

- **The tables are DLSSNR-AMD's** (`linux/data/int4/rdna4`, 28 MB, calibrated on its own set of game frames) and are not
  part of this repository. Point `--int4-tables` or `DLSSNR_INT4_TABLES` at them; the default is a `DLSSNR-AMD`
  checkout beside this one. They hold no weights: the int4 weights are rebuilt from your extracted weights when the
  network is built (a few seconds), by the decoder in `dlssnr/int4.py`, which reproduces all 83 matrices of
  DLSSNR-AMD's installer bit for bit (`tests/test_int4.py`).
- **It matches DLSSNR-AMD's network** to 46.9 to 48.3 dB (the exact network here matches its exact pictures to 48.0 and
  49.8 dB at 1080p and 4K). The remaining difference is float rounding, which an int4 decision amplifies.
- **There is no speed-up.** This emulates the int4 products exactly in float32, and it uses the reference arithmetic, not
  the Triton kernels: 0.51 s a frame at 1080p and 2.3 s at 4K on a 7900 XTX (3.5 and 10.0 GiB), against 0.19 and 0.77 s
  for the exact network. The result is the same on the CPU, bit for bit. The speed of the int4 network is in DLSSNR-AMD.
- The method and the tables are DLSSNR-AMD's (MIT license, copyright mochizuki0323); this is a port of its arithmetic.

## Getting the weights

`nvngx_dlssnr.dll` is NVIDIA's Neural Rendering library, shipped inside NVIDIA's DLSS 5 / NGX packages. The
extractor reads bytes from it (it never loads or runs the library) and accepts either the DLL or a zip that
contains it:

```
dlssnr extract nvngx_dlssnr.dll -o weights.safetensors        # about a second, 141 MB
```

## Running it

```bash
dlssnr run frame.png -o enhanced.png                 # enhance an image
dlssnr run frame.png --side-by-side                  # the input and the result in one picture
dlssnr run frame.png --int4                           # the int4 mixed network of DLSSNR-AMD (see below)
dlssnr run frame.png --intensity 1.5 --style 0 --local-tone 1 --local-structure 1 --skin-structure -1
dlssnr verify --input frame.png --reference nvidia.png -o ours.png --min-psnr 44
dlssnr plan 1920x1080                                # the layers of the network for a frame size
dlssnr benchmark --size 3840x2160 --all              # time a frame; --layers and --stages show where it goes
```

Frames of any size work: the network pads them to the extent it needs and crops the result. `verify` prints
the PSNR and SSIM against the reference, the RMS error, how many pixels are within one 8-bit step, and how
well the edit matches; `--min-psnr` makes it exit non-zero below a threshold, for scripts.

From Python:

```python
from dlssnr.io import Controls, load_image, save_image
from dlssnr.model import NRNet
from dlssnr.weights import Weights

frame = load_image("frame.png")                         # uint8 [height, width, 3]
net = NRNet(Weights.load("weights.safetensors"), frame.shape[1], frame.shape[0])
enhanced = net(frame, Controls(intensity=1.0))          # uint8 [height, width, 3]
save_image("enhanced.png", enhanced)
```

`Controls` has the controls the real library takes, with the defaults NVIDIA's reference frames were made
with: `intensity` 1.0, `style` 0, `local_tone` 1.0, `local_structure` 1.0, `skin_structure` -1.0, `auto_mask`
true. `depth`, `motion`, `history` and `reset` are accepted so that calling code can match the library's
interface, and are ignored: this version computes single frames (every frame as the first frame of a
sequence). `net(frame, controls, capture=True)` also returns every layer's output.

## Searching block pruning combinations

`scripts/ablate.py` searches combinations of shape-preserving blocks to skip, with the full network's
output as the fidelity target. The default budget retains at most 80% of the matrix multiply-adds:

```bash
python scripts/ablate.py --input frames/ --keep-ratio 0.8 --max-evals 400 --out out/ablation.md
```

`--input` accepts one image or a directory of same-size images (non-recursive). An optional `--reference`
accepts a reference image or directory with matching filenames; these scores do not change the ranking.
The search measures all 59 single-block skips, measures a feasible fallback, then explores combinations
with a bounded beam. `--beam-width` (2) and `--expand-top-k` (6) control its breadth. Each unique evaluation
runs every input image; `--max-evals` includes the singles and fallback and must be at least 60. Baseline
inference and final `--top-k` (10) SSIM replays are counted separately.
Timestamped progress is printed immediately during model loading, baseline inference, each combination
and image evaluation, beam expansion, and final SSIM scoring. Completion logs include the current best
feasible result and elapsed time; output is flushed when redirected to a file or piped through `tee`.

The Markdown report lists the best measured combinations, block and layer indices, remaining MACs,
PSNR and SSIM. The sibling JSON includes all evaluated combinations and per-image scores. Reports are
updated throughout the run, including on interruption. The search is heuristic and does not guarantee
the best possible combination; MAC reductions do not directly predict latency reductions. This searches
skip configurations without changing the weights. `net.skip` takes layer indices, not block numbers.

To generate images from the best combination without rerunning the search:

```bash
python scripts/compare_ablation.py --report out/ablation.json --input frames/ \
  --out out/ablation_comparison --device cuda
```

This saves matching PNGs in `baseline/` (complete model), `pruned/` (best dropped blocks), and
`side_by_side/` (left: baseline, right: pruned). `comparison.json` records the block/layer mapping,
MACs and per-image MSE/PSNR/SSIM. The report's controls are reused, and the original weights must match
the ones used for the search (`--weights weights.safetensors` by default). Omit `--input` to replay
the image paths recorded in the report; use it to override paths from another machine. `--report`
also accepts `ablation.md` when its sibling JSON exists. Inputs must match the report's dimensions.
Use a new or empty output directory; completed image pairs and their manifest survive interruption.

## Distilling a pruned model

`scripts/distill.py` restores a fixed pruning mask by training all retained learned weights against the
full model. It uses the default controls and noise seed, batch size one, FP32 master weights and exact
inference-format rounding with surrogate gradients. The original inference path keeps its arithmetic.

Split raw screenshots by game/scene into training and validation directories before preparing targets;
identical source images cannot occur in both sets. Each cache and both sets must have matching frame
dimensions. Images are never resized. Generate caches on the training machine: their manifests reference
the original image paths and verify image and weight hashes. At 1080p, each float32 RGB target takes
about 25 MB before filesystem overhead.

```bash
# Generate full-frame floating-point targets with the complete frozen teacher.
python scripts/distill.py prepare --input data/train --out out/teacher_train --device cuda
python scripts/distill.py prepare --input data/val --out out/teacher_val --device cuda

# Use best.blocks from ablate.py's JSON report; the mask must retain <=80% MACs.
python scripts/distill.py train --pruning-report out/ablation.json \
  --train-cache out/teacher_train --val-cache out/teacher_val --out out/distilled --device cuda

# Resume at the last completed epoch, using the same configuration and cache files.
python scripts/distill.py train --pruning-report out/ablation.json \
  --train-cache out/teacher_train --val-cache out/teacher_val --out out/distilled \
  --device cuda --resume out/distilled/last.pt

# Reload materialized weights through the ordinary inference graph and score them.
python scripts/distill.py evaluate --checkpoint out/distilled/best.safetensors \
  --cache out/teacher_val --out out/distilled/evaluation.json --device cuda
```

`evaluate` also saves a full-resolution side-by-side PNG for **every** validation image. The columns
are **input | cached full-model teacher | distilled student**, from left to right, with no resizing.
For the command above, PNGs are written to `out/distilled/evaluation_images/`; `comparison.json` in
that directory maps each numbered PNG to its input filename and records the column order and status.
The evaluation JSON also includes this comparison manifest. All images reuse the scoring inference
pass, and completed PNGs are preserved if evaluation is interrupted.

Use `--images-out PATH` to choose a new or empty directory, or `--no-images` to produce only metrics.
By default the image directory is `<report stem>_images` beside `--out`; when `--out` is omitted it is
`evaluation_images` beside the checkpoint. Existing nonempty image directories are rejected, so use
a different directory when comparing another checkpoint. Training's per-epoch validation continues
to use the TensorBoard image settings described below.

All commands accept `--weights` (default `weights.safetensors`). The extraction weights are still needed
to construct the inference skeleton and must match the teacher fingerprint. The teacher targets are
computed once, so training does not keep a teacher on the GPU. Repeating `prepare` resumes a compatible
partial cache or reuses a complete one.

Training defaults are 20 epochs, AdamW without weight decay, learning rate `1e-5` with cosine decay to
`1e-6`, gradient accumulation over four frames, gradient clipping at 1.0 and five-epoch early stopping.
The loss is L1 plus 0.1 times the mean horizontal/vertical edge L1. Whole-block non-reentrant activation
checkpointing is enabled; `--no-checkpoint` disables it. A full-frame backward preflight checks gradients
and records memory before optimizer updates; Adam's states allocate additional memory on the first
update. Peak memory for the full run is also recorded. Training is intended for the requested 24GB GPU;
the preflight reports actual capacity failures without changing resolution.

`last.pt` and `best.pt` preserve FP32 masters, optimizer and scheduler state, the pruning mask and data
fingerprints. `best.safetensors` plus `best.json` contain materialized buffers for ordinary `NRNet`:

```python
from dlssnr.distill_train import load_export

net, metadata = load_export("weights.safetensors", "out/distilled/best.safetensors", device="cuda")
enhanced = net(frame)  # the normal uint8 numpy interface; the pruning mask is restored automatically
```

`report.md` and `report.json` compare the pruning-only baseline with every epoch, record per-image
validation scores and memory use, and re-evaluate the best export after loading it into the inference
model. Float PSNR uses the unrounded [0,1] targets; uint8 PSNR and SSIM use the original frame rounding.
The baseline remains the best artifact if training never improves validation error. Logs are flushed
throughout preparation, training and evaluation. Interrupting preparation preserves completed targets;
interrupting training preserves the last completed epoch.

### TensorBoard

Install the optional extra alongside your hardware extra, for example
`uv sync --extra cuda --extra tensorboard`. For an existing environment,
`uv pip install tensorboard` installs just the logging dependency.

```bash
python scripts/distill.py train --pruning-report out/ablation.json \
  --train-cache out/teacher_train --val-cache out/teacher_val \
  --out out/distilled --device cuda --tensorboard

tensorboard --logdir out/distilled/tensorboard
```

Open the address printed by TensorBoard (usually `http://localhost:6006`). Logging is opt-in;
without `--tensorboard`, the dependency is not imported and no event directory is created.

All curves and images use **optimizer updates** as their step axis. `train/loss_step` is the mean
unscaled loss of the actual accumulation group, including a shorter final group. Each update also
records `train/learning_rate` and the **pre-clipping** `train/grad_norm`. Completed epochs record
`train/loss_epoch`, `train/epoch`, `val/loss`, float and uint8 MSE/PSNR, available SSIM, best MSE/epoch,
and `val/bad_epochs`. Validation loss uses the same L1-plus-edge loss on the ordinary inference graph.
The pruning-only baseline is step 0; preflight does not count as training. Infinite PSNR is recorded
as text rather than a finite scalar; unavailable SSIM is omitted.

The Images tab shows the first three validation frames (or all frames if fewer), in fixed manifest
order. Each comparison is **input | teacher | student**, left to right, with each panel at its original
resolution and the filename in the corresponding Text entry. Comparisons are recorded for the
baseline and each epoch under `val/images`, then for the reloaded best export under `final/images`.
Final metrics use `final/*` tags. These full-resolution PNG events can make logs large; image logging
reuses the validation forward passes and does not resize training or evaluation inputs.

Pass `--tensorboard` again when resuming with `--resume out/distilled/last.pt`. The same log directory
is reused, and events beyond the restored checkpoint's optimizer step are hidden so abandoned
partial epochs do not appear twice. Logging can also be enabled when resuming an older checkpoint;
only subsequent work is recorded, without reconstructing historical curves or images. Events flush
every 30 seconds, at completed epochs, and on normal or interrupted shutdown. The existing JSON and
Markdown reports remain available independently of TensorBoard.

## The network

71 blocks, 152 layers, 141 MB of FP8 weights, in a U-Net shape. A frame is padded to a multiple of what its
halvings need; sizes below are for 1920x1080.

| Blocks | Layers | Resolution | Channels |
| --- | --- | --- | --- |
| 0 | pre-block: the frame lifted to features with seeded noise and the conditioning, one Swin layer, 2x2 pool | 1088x1920 to 544x960 | 3 to 32 |
| 1-4 | fused Swin layers, 1 head, windowed attention + MLP; the last pools and widens | 544x960 | 32 to 64 |
| 5-8 | Swin layers, 2 heads | 272x480 | 64 to 128 |
| 9-14 | Swin layers, 4 heads | 136x240 | 128 to 256 |
| 15-22 | Swin layers, 8 heads | 68x120 | 256 to 512 |
| 23-30 | "split Swin" blocks: grouped feed-forward, projection, 16-head shifted-window attention, projection; the last pools | 36x60 | 512 to 1024 |
| 31-38 | vision-transformer blocks: global attention over all tokens, 4x feed-forward | 20x32 | 1024 |
| 39 | decoder input: project to 512 and upsample, merged with the skip from block 30 | 36x60 | 1024 to 512 |
| 40-47 | split Swin blocks again | 36x60 | 512 |
| 48-69 | Swin layers mirroring the encoder, each level starting with an upsample merged with that level's skip | 68x120 up to 544x960 | 512 down to 32 |
| 70 | post-block: features to 3 channels, blended with the frame by the intensity | 1088x1920 | 32 to 3 |

Numerics follow the library, because the weights were trained and calibrated for them. Tensors are float32
that lie exactly on the FP8 e4m3 grid (`dlssnr.quant.q8` rounds to nearest even and saturates at 448, which is
PyTorch's `float8_e4m3fn` cast after a clamp); an e4m3 product has at most 8 significant bits, so products
are exact and accumulation is float32. Where the library keeps FP16 intermediates (activations, softmax
sums, the attention exponential, residual gains) the layer rounds through FP16 at the same points. The
activation is a cubic SiLU, q and k are normalised per head with per-head scales, and the position bias is
baked into the exponent.

## Limits

- Single frames only. NVIDIA's library blends in the previous frame's output through history features and
  motion vectors; those inputs are accepted and not used.
- Not bit-identical to NVIDIA's output (see Results), and the noise the library adds is reproduced from its
  generator for one fixed seed.
- Intensity above 1 uses the library's luminance-ratio form; it has not been compared against the library.
- Speed: 0.2 to 0.8 seconds a frame from 1080p to 4K on a 7900 XTX, 20 to 90 seconds on 16 CPU cores. The
  FP8 rounding is emulated, and the layers outside the fused kernels still run as separate PyTorch
  operations.
- The Triton kernels have only been run on the 7900 XTX (ROCm). They use an AMD instruction, so the CUDA build
  stays on the reference arithmetic.
- The layouts of the weight file were worked out from the library by inspection; a different version of
  `nvngx_dlssnr.dll` than the one this was built against may not extract.

The code is under the MIT license (see `LICENSE`); the network's weights are NVIDIA's and are not covered by it.

This is an independent project. It is not affiliated with, endorsed by, or supported by NVIDIA. DLSS is a
trademark of NVIDIA Corporation. The network's weights are NVIDIA's.
