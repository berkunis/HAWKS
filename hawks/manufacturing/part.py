"""Part and layer domain helpers."""

from __future__ import annotations

from hawks.types import DefectInstance, LayerResult, PartResult


def build_part(part_id: str, layer_defects: list[list[DefectInstance]]) -> PartResult:
    """Construct a PartResult from per-layer defect lists."""
    layers = []
    total = 0
    for i, defects in enumerate(layer_defects):
        layers.append(LayerResult(layer_index=i, defects=defects))
        total += len(defects)
    return PartResult(part_id=part_id, layers=layers, total_defect_count=total)
