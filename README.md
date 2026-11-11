# vfm-quality-control-benchmark

Zero-shot evaluation of vision foundation models (SAM 3, YOLOE) on industrial defect benchmarks (3CAD, VISION).

## Setup

Requires [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just) and an NVIDIA GPU (CUDA 13).

```bash
just setup                # git hooks and virtual environment
cp .env.example .env
```

SAM 3 weights are gated: request access at https://huggingface.co/facebook/sam3, then

```bash
uv run hf auth login      # read token
```

YOLOE weights download on first use to `checkpoints/`; its text encoder (`mobileclip2_b.ts`) to the working directory.

Datasets go in `data/3CAD/` and `data/vision_datasets/`.

## Usage

```bash
uv run python scripts/smoke.py --model sam3 \
  --config src/vfm_quality_control_benchmark/configs/models.yaml \
  --image data/3CAD/3CAD/Aluminum_Camera_Cover/test/scratche/000001.png \
  --prompt scratch
```

`--model yoloe` for YOLOE. Interactive: `notebooks/explorer.ipynb`.