import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

from vfm_quality_control_benchmark.data_types.enums import ModelName
from vfm_quality_control_benchmark.models.factory import Factory
from vfm_quality_control_benchmark.utils.tasks import (
    segm_scores,
    to_coco_results,
)
from vfm_quality_control_benchmark.utils.vision_datasets import (
    image_path,
    load_prompts,
    load_split,
    prompt_to_categories,
)

SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=ModelName)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--limit', type=int, help='first N images by id')
    ap.add_argument('--models-config', required=True, type=Path)
    ap.add_argument('--prompts-config', required=True, type=Path)
    ap.add_argument(
        '--data-root', default=Path('data/vision_datasets'), type=Path
    )
    ap.add_argument('--out', default=Path('outputs/vision'), type=Path)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)  # noqa: NPY002
    torch.manual_seed(SEED)

    coco = load_split(args.data_root, args.dataset, args.split)
    mapping = prompt_to_categories(
        load_prompts(args.prompts_config), args.dataset, coco
    )
    prompts = list(mapping)
    img_ids = sorted(coco.getImgIds())[: args.limit]
    model = Factory.build_model(
        Factory.load_config(args.model, args.models_config)
    )

    results: list[dict] = []
    start = time.perf_counter()
    for i, img_id in enumerate(img_ids, 1):
        path = image_path(
            args.data_root, args.dataset, args.split, coco, img_id
        )
        pred = model.predict(Image.open(path), prompts)
        results += to_coco_results(pred, img_id, mapping)
        if i % 10 == 0 or i == len(img_ids):
            logger.info(f'{i}/{len(img_ids)} images')
    seconds = time.perf_counter() - start

    metrics = segm_scores(coco, results, img_ids)
    logger.info(f'{args.model} {args.dataset}/{args.split}: {metrics}')

    run = f'{args.split}_n{len(img_ids)}'
    out = args.out / args.model.value / args.dataset / run
    out.mkdir(parents=True, exist_ok=True)
    (out / 'predictions.json').write_text(json.dumps(results))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (out / 'run.json').write_text(
        json.dumps(
            {
                'dataset': args.dataset,
                'split': args.split,
                'n_images': len(img_ids),
                'n_images_in_split': len(coco.getImgIds()),
                'image_ids': img_ids,
                'prompt_to_categories': mapping,
                'categories': coco.loadCats(coco.getCatIds()),
                'model': model.metadata(),
                'seed': SEED,
                'seconds': round(seconds, 1),
                'seconds_per_image': round(seconds / len(img_ids), 3),
                'python': platform.python_version(),
            },
            indent=2,
            default=str,
        )
    )
    logger.info(f'saved {out}')


if __name__ == '__main__':
    main()
