from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pycocotools import mask as mask_utils

if TYPE_CHECKING:
    from vfm_quality_control_benchmark.data_types.models import (
        InstancePredictions,
    )


def to_anomaly_map(pred: InstancePredictions) -> tuple[np.ndarray, float]:
    """3CAD: per-pixel max mask score (0 where nothing is predicted); image score = map max."""
    h, w = pred.image_size
    amap = np.zeros((h, w), dtype=np.float32)
    for mask, score in zip(pred.masks, pred.scores, strict=True):
        np.maximum(
            amap, np.where(mask, score, 0.0).astype(np.float32), out=amap
        )
    return amap, float(amap.max(initial=0.0))


def to_coco_results(
    pred: InstancePredictions,
    image_id: int,
    prompt_to_category: dict[str, int],
) -> list[dict]:
    """VISION: COCO instance-segmentation results (RLE masks), one dict per instance.

    `prompt_to_category` maps every prompt string to a COCO category_id; it comes
    from the prompts config, so generic prompts (e.g. "scratch") map to an explicit class.
    """

    results = []
    for mask, box, score, pid in zip(
        pred.masks, pred.boxes, pred.scores, pred.prompt_ids, strict=True
    ):
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')
        x1, y1, x2, y2 = (float(v) for v in box)
        results.append(
            {
                'image_id': int(image_id),
                'category_id': int(prompt_to_category[pred.prompts[pid]]),
                'segmentation': rle,
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': float(score),
            }
        )
    return results
