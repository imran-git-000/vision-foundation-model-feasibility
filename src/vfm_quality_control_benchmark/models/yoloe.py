from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import ultralytics
from ultralytics import YOLOE

from vfm_quality_control_benchmark.data_types.models import (
    InstancePredictions,
    PredictSettings,
)

if TYPE_CHECKING:
    from PIL import Image


class YoloeSegmenter:
    """YOLOE text-prompted segmentation through Ultralytics.

    All prompts run in one forward pass. NMS is per prompt (not class-agnostic),
    so each prompt's instances are kept independently, as with SAM 3.
    """

    def __init__(
        self,
        *,
        model_id: str = 'yoloe-26x-seg.pt',
        revision: str = 'main',
        device: str | None = None,
        dtype: str = 'float32',
        image_size: int | None = None,
        settings: PredictSettings | None = None,
    ) -> None:
        if revision != 'main':
            raise ValueError('YOLOE weights are a file; pin them by model_id')
        if dtype != 'float32':
            raise ValueError('the YOLOE wrapper runs in float32 only')
        self.settings = settings or PredictSettings()
        if self.settings.mask_threshold != 0.5:  # noqa: PLR2004
            raise ValueError('Ultralytics binarises masks at a fixed 0.5')
        self.model_id = model_id
        self.dtype_name = dtype
        self.image_size = image_size or 640
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = YOLOE(model_id)
        self.weights_path = Path(self.model.ckpt_path or model_id)
        self.weights_sha256 = hashlib.sha256(
            self.weights_path.read_bytes()
        ).hexdigest()
        self._prompts: list[str] | None = None

    def predict(
        self, image: Image.Image, prompts: list[str]
    ) -> InstancePredictions:
        if not prompts:
            raise ValueError('predict needs at least one prompt')
        if prompts != self._prompts:  # text embedding is the slow part
            self.model.set_classes(prompts)
            self._prompts = list(prompts)
        image = image.convert('RGB')
        size_hw = (image.height, image.width)
        s = self.settings

        r = self.model.predict(
            image,
            imgsz=self.image_size,
            conf=s.score_threshold,
            max_det=s.max_detections,
            agnostic_nms=False,
            retina_masks=True,  # masks at native image resolution
            device=self.device,
            verbose=False,
        )[0]
        if r.masks is None or len(r.boxes) == 0:
            return InstancePredictions.empty(prompts, size_hw)
        return InstancePredictions(
            masks=r.masks.data.bool().cpu().numpy(),
            boxes=r.boxes.xyxy.float().cpu().numpy().astype(np.float32),
            scores=r.boxes.conf.float().cpu().numpy().astype(np.float32),
            prompt_ids=r.boxes.cls.cpu().numpy().astype(np.int64),
            prompts=list(prompts),
            image_size=size_hw,
        ).top_k(s.max_detections)

    def metadata(self) -> dict[str, Any]:
        return {
            'model': 'yoloe',
            'backend': 'ultralytics',
            'model_id': self.model_id,
            'weights_path': str(self.weights_path),
            'weights_sha256': self.weights_sha256,
            'dtype': self.dtype_name,
            'model_input_size': self.image_size,
            'device': self.device,
            'settings': self.settings.model_dump(),
            'versions': {
                'torch': torch.__version__,
                'ultralytics': ultralytics.__version__,
            },
        }
