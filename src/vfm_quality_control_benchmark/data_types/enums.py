from enum import StrEnum


class ModelName(StrEnum):
    """Models under evaluation. Values match the keys in configs/models.yaml."""

    SAM3 = 'sam3'
    YOLOE = 'yoloe'
