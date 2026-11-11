from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import yaml

from vfm_quality_control_benchmark.data_types.enums import ModelName
from vfm_quality_control_benchmark.data_types.models import ModelConfig
from vfm_quality_control_benchmark.models.sam3 import Sam3Segmenter
from vfm_quality_control_benchmark.models.yoloe import YoloeSegmenter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from vfm_quality_control_benchmark.models.base import (
        TextPromptedSegmenter,
    )


class Factory:
    """Builds model wrappers from a validated `ModelConfig`."""

    _registry: ClassVar[
        dict[ModelName, Callable[..., TextPromptedSegmenter]]
    ] = {
        ModelName.SAM3: Sam3Segmenter,
        ModelName.YOLOE: YoloeSegmenter,
    }

    @staticmethod
    def load_config(name: ModelName, path: Path) -> ModelConfig:
        """Read and validate the entry for `name` from the models YAML at `path`."""
        entries = yaml.safe_load(path.read_text(encoding='utf-8'))
        if name.value not in entries:
            raise KeyError(f'{path} has no entry for {name.value!r}')
        return ModelConfig.model_validate(entries[name.value])

    @classmethod
    def build_model(cls, config: ModelConfig) -> TextPromptedSegmenter:
        """Instantiate the wrapper registered for `config.name`."""
        model_cls = cls._registry.get(config.name)
        if model_cls is None:
            raise NotImplementedError(
                f'no wrapper registered for {config.name}'
            )
        return model_cls(settings=config.predict, **config.init.model_dump())
