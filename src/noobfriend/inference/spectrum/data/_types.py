"""Spectrum data type aliases.

Internal: both are ``Literal`` unions of the accepted strings, so callers
write the string itself rather than importing the alias.
"""

from __future__ import annotations

from typing import Literal


type DispersionAxis = Literal["row", "column"]
type NoiseSource = Literal["manual", "continuum", "background"]
