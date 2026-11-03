"""Workspace fingerprints: what a saved fit must match to be reattached.

Two independent SHA-256 digests (T11c.3):

- data: every frame's arrays, mask, fitted pixels and resolving power, the
  noise declarations, the window, ``z`` and the unit;
- model: the parameter table (names, kinds, priors, fixed values, pooling),
  the line quantities' expressions, line shapes and the continuum form.

A frame's own array digest (obs, flux, error) also identifies it for the
"do not fit the same pixels twice" check of joint priors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from noobfriend.inference.spectrum.workspace._handles import ContinuumHandle, LineHandle
from noobfriend.inference.spectrum.workspace._table import JointBlock, ParameterTable


def frame_digest(frame: Any) -> str:
    """Digest of one frame's ``obs``, ``flux`` and ``error`` arrays."""
    hasher = hashlib.sha256()
    for array in (frame.obs, frame.flux, frame.error):
        _update_array(hasher, array)
    return hasher.hexdigest()


def data_fingerprint(
    spectrum: Any,
    fitted: Mapping[str, np.ndarray],
    window: Iterable[tuple[float, float]] | None,
) -> str:
    """Digest of the data side of a workspace."""
    hasher = hashlib.sha256()
    _update_text(hasher, f"unit={spectrum.unit};z={spectrum.z!r};window={window!r}")
    for frame_id, frame in spectrum.frames.items():
        _update_text(hasher, f"frame={frame_id};noise={frame.noise!r}")
        for array in (frame.obs, frame.flux, frame.error, frame.mask, fitted[frame_id]):
            _update_array(hasher, array)
        resolving_power = frame.resolving_power
        if isinstance(resolving_power, tuple):
            for array in resolving_power:
                _update_array(hasher, array)
        else:
            _update_text(hasher, f"R={resolving_power!r}")
    return hasher.hexdigest()


def model_fingerprint(
    table: ParameterTable,
    lines: Iterable[LineHandle],
    continuum: ContinuumHandle,
    joint_blocks: Iterable[JointBlock],
) -> str:
    """Digest of the model side of a workspace."""
    hasher = hashlib.sha256()
    for row in table.values():
        _update_text(
            hasher,
            f"{row.name}|{row.kind}|{row.prior!r}|{row.value!r}|{row.pooled_from}|"
            f"{row.log_pooled}|{row.domain.name}",
        )
    for quantity in table.quantities.values():
        _update_text(hasher, f"{quantity.name}={quantity.expression.text()}")
    for line in lines:
        _update_text(
            hasher,
            f"line={line.name}|{line.profile}|{line.contribution}|{line.nominal!r}|"
            f"{line.kernels}|{dict(line.lsf_fwhm_kms)!r}",
        )
    _update_text(
        hasher,
        f"continuum={continuum.kind}|{continuum.order}|{continuum.lambda_0!r}|"
        f"{continuum.sharing}",
    )
    for block in joint_blocks:
        _update_text(hasher, f"joint={block.parameters}|{block.members}")
        _update_array(hasher, block.mean)
        _update_array(hasher, block.covariance)
    return hasher.hexdigest()


def _update_text(hasher: Any, text: str) -> None:
    hasher.update(text.encode())
    hasher.update(b"\0")


def _update_array(hasher: Any, array: Any) -> None:
    values = np.ascontiguousarray(array)
    hasher.update(f"{values.dtype.str}{values.shape}".encode())
    hasher.update(values.tobytes())
