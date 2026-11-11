# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research question: can general vision foundation models find real manufacturing defects without defect-specific training?

This repo evaluates **SAM 3** and **YOLOE** (open-vocabulary YOLO) zero-shot on two industrial defect benchmarks, **3CAD** and the **VISION Datasets**. The output is a results table comparing both models against the published baselines, plus qualitative examples of successes and failures.

Current scope is zero-shot evaluation only. Fine-tuning and building a dedicated model are later stages; do not start them unless asked.

## Data

Both datasets live under `data/`, one folder each, and each folder contains the original paper as a PDF.

- **Read the paper in each dataset folder before writing any code for that dataset.** It defines the splits, the annotation format and the official metrics.
- Inspect the actual folder layout and annotation format first (`ls`, open a few annotation files). Do not assume the layout from the papers or from this file; if something here disagrees with what is on disk, trust the disk and tell me.
- Treat `data/` as read-only. Never modify, move, rename or delete anything in it. Write derived files (converted annotations, caches, predictions) to `outputs/`.
- Never commit dataset files, model weights or prediction dumps. Keep them in `.gitignore`.

### 3CAD (AAAI 2025)

- 8 part categories from real 3C production lines: ACC, AI, AMF, ANMF, ANI, AP, CS, IS.
- 27,039 images: 10,493 training images (all normal) and 16,546 test images (5,084 normal, 11,462 defective), with pixel-level masks.
- Designed for unsupervised anomaly detection, so the training split has no defects. Zero-shot evaluation uses the **test split** only.
- Defects are tiny: most cover under 0.5% of the image. The paper resizes to 256×256; that can erase small defects, so evaluate at native resolution and record the resolution used.
- Official metrics: image-level AUROC (I-AUROC), pixel-level AUROC (P-AUROC), pixel-level PRO (P-PRO), pixel-level AP. Report per category and the mean.
- Published reference (CFRG, Tables 3–5 of the paper): 93.4 P-AUROC, 86.5 I-AUROC, 82.0 P-PRO, 17.6 AP.

On disk (checked 2026-11-10):

- Paper: `data/3CAD/paper.pdf`. Dataset root: `data/3CAD/3CAD/<Category>/{train,test,ground_truth}/<defect>/NNNNNN.png` (MVTec-style; `test/good` has no masks). `data/3CAD/DTD` and `data/3CAD/MvtecAD` are auxiliary datasets shipped alongside and are not part of the 3CAD benchmark.
- Folders use full names, not the paper's abbreviations: ACC=`Aluminum_Camera_Cover`, AI=`Aluminum_Ipad`, AMF=`Aluminum_Middle_Frame`, ANI=`Aluminum_New_Ipad`, ANMF=`Aluminum_New_Middle_Frame`, AP=`Aluminum_Pc`, CS=`Copper_Stator`, IS=`Iron_Stator`.
- **Defect folders come in duplicate pairs**: a pinyin name and an English name holding byte-identical images (e.g. `aotu`=`uneven`, `huashang`=`scratche`, `yashang`=`bruise`, `zangwu`=`dirt`). The test split has 27,620 PNG files but only 16,496 unique by MD5, against 16,546 in the paper. Deduplicate by content hash before scoring, or every metric double-counts. Normal (5,084) and train (10,493) counts match the paper. Also present: `Multiple-defects` folders and typos like `scratche`.
- Use the English folder name as the prompt, never the pinyin one.
- Test images are **JPEG files with a `.png` extension**. Ground-truth masks are real 8-bit grayscale PNGs. Open both by content (PIL does this), not by extension.
- Image sizes vary within each category, up to 1024 px on the long side. SAM 3 runs at 1008 px internally, so it sees 3CAD at roughly native resolution. Some VISION datasets go up to about 3000 px (PCB_1) and get downscaled about 3×.


### VISION Datasets (arXiv 2306.07890)

- 14 datasets: Cable, Capacitor, Casting, Console, Cylinder, Electronics, Groove, Hemisphere, Lens, PCB_1, PCB_2, Ring, Screw, Wood. 44 defect types.
- Instance-segmentation annotations in train, validation and test splits. Check whether test annotations are actually present on disk; if not, evaluate on validation and say so in the results.
- Official metric: `0.5 * mAP + 0.5 * mAR(max=100)` for instance segmentation, computed per dataset and averaged over the 14 datasets with equal weight. Also report mAP and mAR separately.

On disk (checked 2026-11-10):

- Paper: `data/vision_datasets/VISION Datasets.pdf`; dataset card: `data/vision_datasets/README.md`.
- Each dataset is a `<Name>.tar.gz` archive. Because `data/` is read-only, extract to `outputs/` (or read members straight from the tarball). Archives contain `<Name>/{train,val,inference}/_annotations.coco.json` plus `.jpg` images.
- Archives include macOS `._*` AppleDouble files next to every image; skip them.
- **There are no test labels**: `inference/` lists test images plus discarded ones, with zero annotations. Evaluate on `val` and say so in the results. That also means `val` cannot be used for prompt or threshold selection without contaminating the reported numbers. Ask before making any such choice.
- Class names vary from informative (Cable: `break`, `thunderbolt`) to generic (Screw: `defect`).

## Models

- **SAM 3**: prompt with text (defect names). Mask scores are the confidence used for ranking and for building anomaly maps. Runs through Hugging Face Transformers (`Sam3Model` / `Sam3Processor`, checkpoint `facebook/sam3`), wrapped in `src/vfm_quality_control_benchmark/models/sam3.py`. Check the current Transformers docs for the API before changing it; do not rely on memory. The checkpoint is gated: if a download fails for access reasons, stop and tell me.
- **YOLOE**: open-vocabulary segmentation variant, prompted with the same text prompts as SAM 3 so the comparison is fair. Do not use Ultralytics. The backend is not chosen yet; ask before picking one.
- Every model wrapper implements the same interface (`models/base.py`: `predict(image, prompts) -> InstancePredictions`, `metadata()`). Dataset formats (3CAD anomaly maps, VISION COCO results) are built from that in `tasks.py`, never inside a wrapper.
- Zero-shot means no training or tuning on either dataset. Do not select prompts or thresholds on the test split; use VISION validation for any such choice and report it.

### Prompts

- Use the defect class names from each dataset's annotations as prompts, plus a generic set (`defect`, `scratch`, `dent`, `crack`, `stain`) for classes with uninformative names (for example `Defect1`, `Defect-A`).
- Keep all prompts in one config file, never hard-coded in scripts, and save the prompts used alongside every result.

### Turning predictions into metrics

- VISION: export predictions in COCO instance-segmentation format and score with `pycocotools`.
- 3CAD: build a per-pixel anomaly map as the maximum mask score over all predicted masks (0 where nothing is predicted); the image score is the maximum of that map. Compute the four metrics from those.

## Repo layout

```
data/            datasets and papers (read-only)
src/vfm_quality_control_benchmark/   package: dataset loaders, model wrappers, metrics
src/vfm_quality_control_benchmark/configs/   prompts and run settings (YAML, read via importlib.resources)
scripts/         entry points (one per model/dataset run)
outputs/         predictions, metrics, figures (git-ignored)
results/         final tables (CSV + Markdown), committed
```

So far only `data/` and a stub package exist (`main()` prints a greeting); the rest is planned.

## Environment and commands

The project is a `uv` project (`pyproject.toml`, `uv_build` backend, src layout). Python is **3.14** (`.python-version`), by decision; do not lower it. PyTorch comes from the CUDA 13.0 index (`pytorch-cu130`) configured in `pyproject.toml` (the GPU is an RTX 5000 Ada with 32 GB). There is no system `python` on PATH, so always go through uv:

```
uv sync                                  # create .venv from pyproject/uv.lock
uv add <pkg>==<version>                  # add a pinned dependency
uv run python scripts/<script>.py        # run an entry point
uv run python -m pytest tests/test_x.py::test_name   # single test, once pytest is added
uv export --format requirements-txt > requirements.txt   # produce the pinned requirements.txt
```

No test or lint tooling is configured yet.

## Working rules

- Start every new run on a small subset (one 3CAD category, one VISION dataset, about 50 images) to confirm the pipeline end to end before launching full runs.
- Never invent, estimate or fill in a number. Every figure in `results/` must come from a saved metrics file in `outputs/`. If a run failed or was partial, write that instead of a number.
- Poor zero-shot results are an expected and valid outcome. Report them as they are; do not tune until they look better.
- Validate metric code before trusting it: all-zero predictions should give chance-level AUROC, and ground-truth masks fed back as predictions should give perfect scores.
- Fix random seeds and log model name, weights file, image resolution, prompts and package versions with each run.
- Save 10–20 qualitative overlays (image, ground truth, prediction) per dataset, including failures.
- Ask before long or expensive jobs (full-dataset runs, large downloads) and before adding a heavy dependency.
- Python with `torch`, `transformers`, `pycocotools`, `scikit-learn`, `numpy`, `opencv-python`. Pin versions in `pyproject.toml` / `uv.lock` and export them to `requirements.txt`.

## Deliverables

1. `results/zero_shot.md`: table of SAM 3 and YOLOE on both datasets (per category/dataset and mean), next to the published 3CAD reference.
2. `results/figures/`: qualitative overlays.
3. A short note on where each model fails (defect size, contrast, prompt wording) and whether few-shot tuning looks worthwhile.
