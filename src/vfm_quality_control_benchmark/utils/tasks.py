from __future__ import annotations

import contextlib
import copy
import io
from typing import TYPE_CHECKING

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval

if TYPE_CHECKING:
    from pycocotools.coco import COCO

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
    prompt_to_categories: dict[str, list[int]],
) -> list[dict]:
    """VISION: COCO instance-segmentation results (RLE masks).

    `prompt_to_categories` maps every prompt to the COCO category ids it stands
    for (from the prompts config). An instance is emitted once per category, so
    a generic prompt shared by indistinguishable classes is scored for each.
    """
    results = []
    for mask, box, score, pid in zip(
        pred.masks, pred.boxes, pred.scores, pred.prompt_ids, strict=True
    ):
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')
        x1, y1, x2, y2 = (float(v) for v in box)
        for category_id in prompt_to_categories[pred.prompts[pid]]:
            results.append(  # noqa: PERF401
                {
                    'image_id': int(image_id),
                    'category_id': int(category_id),
                    'segmentation': rle,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': float(score),
                }
            )
    return results


def segm_scores(
    gt: COCO, results: list[dict], img_ids: list[int]
) -> dict[str, float]:
    """mAP@[.5:.95] and mAR@100 (all areas) on `img_ids`, plus the VISION score.

    Restricting to `img_ids` matters for subset runs: images that were not
    predicted would otherwise count as missed defects.
    """
    if not results:  # loadRes rejects an empty list; nothing found = 0
        return {'mAP': 0.0, 'mAR': 0.0, 'score': 0.0}
    with contextlib.redirect_stdout(io.StringIO()):  # COCOeval prints tables
        dt = gt.loadRes(copy.deepcopy(results))  # loadRes mutates its input
        ev = COCOeval(gt, dt, iouType='segm')
        ev.params.imgIds = list(img_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    m_ap, m_ar = float(ev.stats[0]), float(ev.stats[8])
    return {'mAP': m_ap, 'mAR': m_ar, 'score': 0.5 * m_ap + 0.5 * m_ar}
