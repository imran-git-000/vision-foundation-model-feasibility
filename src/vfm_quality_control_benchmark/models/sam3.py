from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from transformers import Sam3Model, Sam3Processor

from vfm_quality_control_benchmark.data_types.models import (
    InstancePredictions,
    PredictSettings,
)

if TYPE_CHECKING:
    from PIL import Image


class Sam3Segmenter:
    def __init__(
        self,
        *,
        model_id: str = 'facebook/sam3',
        revision: str = 'main',
        device: str | None = None,
        dtype: str = 'float32',
        image_size: int | None = None,
        settings: PredictSettings | None = None,
    ) -> None:
        if image_size not in (None, 1008):
            raise ValueError('SAM 3 runs at a fixed 1008 px input size')
        self.model_id = model_id
        self.revision = revision
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.dtype_name = dtype
        dtypes: dict[str, torch.dtype] = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
        }
        self.dtype = dtypes[dtype]
        self.settings = settings or PredictSettings()

        self.processor = Sam3Processor.from_pretrained(
            model_id, revision=revision
        )
        self.model = Sam3Model.from_pretrained(
            model_id, revision=revision, torch_dtype=self.dtype
        )
        self.model.to(self.device).eval()
        self.revision_resolved = self._cached_commit(model_id, revision)

    @torch.inference_mode()
    def predict(
        self, image: Image.Image, prompts: list[str]
    ) -> InstancePredictions:
        if not prompts:
            raise ValueError('predict needs at least one prompt')
        image = image.convert('RGB')
        size_hw = (image.height, image.width)
        s = self.settings

        img_inputs = self.processor(images=image, return_tensors='pt')
        pixel_values = img_inputs.pixel_values.to(
            self.device, dtype=self.dtype
        )
        target_sizes = img_inputs['original_sizes'].tolist()
        vision_embeds = self.model.get_vision_features(
            pixel_values=pixel_values
        )

        parts: list[InstancePredictions] = []
        for prompt_id, prompt in enumerate(prompts):
            text_inputs = self.processor(text=prompt, return_tensors='pt').to(
                self.device
            )
            outputs = self.model(vision_embeds=vision_embeds, **text_inputs)
            res = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=s.score_threshold,
                mask_threshold=s.mask_threshold,
                target_sizes=target_sizes,
            )[0]
            n = len(res['scores'])
            if n == 0:
                continue
            part = InstancePredictions(
                masks=res['masks'].bool().cpu().numpy(),
                boxes=res['boxes'].float().cpu().numpy().astype(np.float32),
                scores=res['scores'].float().cpu().numpy().astype(np.float32),
                prompt_ids=np.full(n, prompt_id, dtype=np.int64),
                prompts=list(prompts),
                image_size=size_hw,
            )
            # Bound memory before merging: no prompt can contribute more than the final cap.
            parts.append(part.top_k(s.max_detections))

        if not parts:
            return InstancePredictions.empty(prompts, size_hw)
        return InstancePredictions.concat(parts, prompts).top_k(
            s.max_detections
        )

    def metadata(self) -> dict[str, Any]:
        return {
            'model': 'sam3',
            'backend': 'transformers',
            'model_id': self.model_id,
            'revision_requested': self.revision,
            'revision_resolved': self.revision_resolved,
            'dtype': self.dtype_name,
            'model_input_size': {
                'height': self.processor.image_processor.size.height,
                'width': self.processor.image_processor.size.width,
            },
            'device': str(self.device),
            'device_name': torch.cuda.get_device_name(self.device)
            if self.device.type == 'cuda'
            else 'cpu',
            'settings': {
                'score_threshold': self.settings.score_threshold,
                'mask_threshold': self.settings.mask_threshold,
                'max_detections': self.settings.max_detections,
            },
            'versions': {
                'torch': torch.__version__,
                'transformers': transformers.__version__,
            },
        }

    @staticmethod
    def _cached_commit(model_id: str, revision: str) -> str | None:
        """Commit hash of the snapshot actually loaded, read from the HF cache path."""
        path = try_to_load_from_cache(
            model_id, 'config.json', revision=revision
        )
        return Path(path).parent.name if isinstance(path, str) else None
