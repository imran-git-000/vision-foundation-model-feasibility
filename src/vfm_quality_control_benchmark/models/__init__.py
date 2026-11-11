from .base import InstancePredictions, PredictSettings, TextPromptedSegmenter
from .sam3 import Sam3Segmenter


def build_model(cfg: dict) -> TextPromptedSegmenter:
    """Build a model wrapper from one entry of configs/models.yaml."""
    settings = PredictSettings(**cfg.get('predict', {}))
    name = cfg['name']
    if name == 'sam3':
        return Sam3Segmenter(settings=settings, **cfg.get('init', {}))
    raise ValueError(f'unknown model {name!r}')


__all__ = [
    'InstancePredictions',
    'PredictSettings',
    'TextPromptedSegmenter',
    'build_model',
]
