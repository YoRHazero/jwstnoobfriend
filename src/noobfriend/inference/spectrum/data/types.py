"""Spectrum data type aliases."""

from __future__ import annotations

from typing import Literal


type DispersionAxis = Literal["row", "column"]
type NoiseSource = Literal["manual", "continuum", "background"]
