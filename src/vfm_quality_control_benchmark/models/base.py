from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from PIL import Image

    from vfm_quality_control_benchmark.data_types.models import (
        InstancePredictions,
        PredictSettings,
    )


class TextPromptedSegmenter(Protocol):
    """What every model wrapper provides."""

    settings: PredictSettings

    def predict(
        self, image: Image.Image, prompts: list[str]
    ) -> InstancePredictions:
        """Run every prompt on the image and return the merged, top-k instances."""
        ...

    def metadata(self) -> dict[str, Any]:
        """Everything needed to reproduce a run: model id, weights revision, dtype, resolution, versions."""
        ...
