from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from vfm_quality_control_benchmark.data_types.enums import ModelName


@dataclass
class InstancePredictions:
    """All instances found in one image, across all prompts.

    Args:
        prompts: list of prompt strings, in the order they were run.
        masks:  (N, H, W) bool, at the image's native resolution.
        boxes:  (N, 4) float32, xyxy in absolute pixels.
        scores: (N,) float32 confidence, used for ranking and anomaly maps.
        prompt_ids: (N,) int, index into `prompts` of the prompt that produced each instance.
    """

    masks: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    prompt_ids: np.ndarray
    prompts: list[str]
    image_size: tuple[int, int]  # (H, W)

    @classmethod
    def empty(
        cls, prompts: list[str], image_size: tuple[int, int]
    ) -> InstancePredictions:
        h, w = image_size
        return cls(
            masks=np.zeros((0, h, w), dtype=bool),
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            prompt_ids=np.zeros((0,), dtype=np.int64),
            prompts=list(prompts),
            image_size=image_size,
        )

    def __len__(self) -> int:
        return len(self.scores)

    def top_k(self, k: int) -> InstancePredictions:
        """Keep the k highest-scoring instances (sorted by descending score)."""
        order = np.argsort(-self.scores, kind='stable')[:k]
        return InstancePredictions(
            masks=self.masks[order],
            boxes=self.boxes[order],
            scores=self.scores[order],
            prompt_ids=self.prompt_ids[order],
            prompts=self.prompts,
            image_size=self.image_size,
        )

    @classmethod
    def concat(
        cls, parts: list[InstancePredictions], prompts: list[str]
    ) -> InstancePredictions:
        """Merge per-prompt predictions. `prompt_ids` in each part must already index `prompts`."""
        if not parts:
            raise ValueError('concat needs at least one part')
        return cls(
            masks=np.concatenate([p.masks for p in parts]),
            boxes=np.concatenate([p.boxes for p in parts]),
            scores=np.concatenate([p.scores for p in parts]),
            prompt_ids=np.concatenate([p.prompt_ids for p in parts]),
            prompts=list(prompts),
            image_size=parts[0].image_size,
        )


class Config(BaseModel):
    """Base for config models: immutable, and unknown keys are an error."""

    model_config = ConfigDict(extra='forbid', frozen=True)


class PredictSettings(Config):
    """Inference settings. Fixed in config, never tuned on evaluation data."""

    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description='Keep low-confidence instances; ranking metrics need them.',
    )
    mask_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description='Binarisation of per-pixel mask probabilities.',
    )
    max_detections: int = Field(
        default=100,
        gt=0,
        description='Per image, after merging prompts (COCO maxDets=100).',
    )


class ModelInitConfig(Config):
    """How to load a model: which weights, on which device, in which precision."""

    model_id: str
    revision: str = 'main'
    device: str | None = None
    dtype: Literal['float32', 'bfloat16', 'float16'] = 'float32'


class ModelConfig(Config):
    """One entry of configs/models.yaml."""

    name: ModelName
    init: ModelInitConfig
    predict: PredictSettings = PredictSettings()
