"""Lazy dependency provider for the v2 medical tool layer."""

from __future__ import annotations

from src.tools.medical_tools import MedicalToolbox


_toolbox: MedicalToolbox | None = None


def get_toolbox() -> MedicalToolbox:
    global _toolbox
    if _toolbox is None:
        _toolbox = MedicalToolbox()
    return _toolbox
