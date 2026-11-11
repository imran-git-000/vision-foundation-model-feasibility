"""VISION Datasets: COCO instance-segmentation splits on disk.

Layout: <root>/<dataset>/<split>/_annotations.coco.json plus images.
Images are read through the JSON `file_name`s; folders also hold macOS
`._*` files, which are never listed in the JSON.
"""

from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING

import yaml
from pycocotools.coco import COCO

from vfm_quality_control_benchmark.data_types.models import PromptsConfig

if TYPE_CHECKING:
    from pathlib import Path


def load_split(root: Path, dataset: str, split: str = 'val') -> COCO:
    """Ground truth of one split as a pycocotools COCO object."""
    path = root / dataset / split / '_annotations.coco.json'
    with contextlib.redirect_stdout(io.StringIO()):  # COCO() prints progress
        return COCO(str(path))


def image_path(
    root: Path, dataset: str, split: str, coco: COCO, img_id: int
) -> Path:
    return root / dataset / split / coco.imgs[img_id]['file_name']


def load_prompts(path: Path) -> PromptsConfig:
    return PromptsConfig.model_validate(
        yaml.safe_load(path.read_text(encoding='utf-8'))
    )


def prompt_to_categories(
    prompts: PromptsConfig, dataset: str, coco: COCO
) -> dict[str, list[int]]:
    """Map each prompt to the COCO category ids it stands for.

    The config's class names must match the dataset's categories exactly.
    """
    per_class = prompts.vision[dataset]
    by_name = {c['name']: c['id'] for c in coco.loadCats(coco.getCatIds())}
    if set(per_class) != set(by_name):
        raise ValueError(
            f'{dataset}: prompt config classes {sorted(per_class)} '
            f'!= dataset classes {sorted(by_name)}'
        )
    mapping: dict[str, list[int]] = {}
    for name, class_prompts in per_class.items():
        for prompt in class_prompts:
            mapping.setdefault(prompt, []).append(by_name[name])
    return mapping
