import argparse
import json
from pathlib import Path

from loguru import logger
from PIL import Image

from vfm_quality_control_benchmark.data_types.enums import ModelName
from vfm_quality_control_benchmark.models.factory import Factory
from vfm_quality_control_benchmark.utils.tasks import to_anomaly_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, type=ModelName)
    ap.add_argument('--image', required=True, type=Path)
    ap.add_argument('--prompt', action='append', required=True)
    ap.add_argument('--config', required=True, type=Path)
    args = ap.parse_args()

    model = Factory.build_model(Factory.load_config(args.model, args.config))
    pred = model.predict(Image.open(args.image), args.prompt)
    _, image_score = to_anomaly_map(pred)

    logger.info(json.dumps(model.metadata(), indent=2, default=str))
    logger.info(
        f'image {pred.image_size}, {len(pred)} instances, image score {image_score:.4f}'
    )
    for i in range(min(len(pred), 10)):
        area = int(pred.masks[i].sum())
        logger.info(
            f'  {pred.prompts[pred.prompt_ids[i]]:<12} score={pred.scores[i]:.4f} area={area}px'
        )


if __name__ == '__main__':
    main()
