"""Shared recipe model for the ``reduce`` codegen.

A recipe declares which pipeline to run (``pipeline`` = ``clear`` / ``grism``),
which NooBox files to reduce (``[select]``) and, per pipeline step, whether to
skip it (``skip``) and which stage to save the frame after it as (``save_as``).

The pipeline-specific bits -- the ordered step *chains*, their starter
``scaffold`` and their code ``render`` -- live in the per-pipeline
``_reduce_<pupil>_stage<n>.py`` / ``_codegen_<pupil>_stage<n>.py`` modules and are
wired together by :mod:`noobfriend.cli.reduce._registry`. Only the data model and
the chain-ordering helper are here, shared across pipelines.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

_Criterion = str | list[str] | None


@dataclass(frozen=True)
class StepSpec:
    """One pipeline step: its recipe key and (where applicable) the code it maps to.

    ``custom`` names a :mod:`noobfriend.reduction` function called on
    ``(data, err, dq)``; ``jwst`` is a dotted ``jwst`` ``Step`` class path called
    via ``.call(model)``. A step may set neither when its rendering is bespoke
    (e.g. the grism master-sky and template-background steps), in which case the
    pipeline's own renderer emits its code.
    """

    name: str
    title: str
    custom: str | None = None
    jwst: str | None = None


class Selection(BaseModel):
    """Which NooBox books a recipe applies to (forwarded to ``NooBox.select``)."""

    stage: str
    pupil: _Criterion = None
    filter: _Criterion = None
    detector: _Criterion = None


class StepConfig(BaseModel):
    """Per-step recipe controls: skip the step, and/or save the frame after it."""

    skip: bool = False
    save_as: str | None = None


class GrismOptions(BaseModel):
    """Grism-pipeline knobs for the per-module sky-residual template (ignored by clear)."""

    scalar: bool = True
    downsample: int = 4
    smooth_sigma: float = 24.0


class Stage3Options(BaseModel):
    """Stage-3 mosaicking knobs: grouping, unified output grid and astrometry.

    ``group_by`` are the :class:`~noobfriend.navigation.NooBook` fields whose
    distinct combinations define one mosaic (``["observation", "filter"]`` =
    one per field per band). The output scale is per channel --
    ``pixel_scale_sw`` for short-wave detectors (``nrc[ab][1-4]``) and
    ``pixel_scale_lw`` for long-wave (``nrc[ab]long``); set them equal for one
    unified grid. Each mosaic is tiled into evenly-sized ``tile_size`` cells with
    a ``tile_overlap`` border, and tiles whose real footprint coverage is below
    ``min_coverage`` are dropped. ``obs_epoch`` is the decimal-year target for
    GAIA proper-motion propagation and ``star_fwhm_px`` the expected
    point-source FWHM for image-source detection. ``in_memory`` (default false)
    spills the skymatch / outlier / resample models to disk -- safe on the large
    fields stage-3 targets; set true to keep them resident for speed on small
    ones. Per-group astrometry sidecars go under ``work_dir`` and are deleted
    afterwards when ``clean_work`` is set.
    """

    group_by: list[str] = ["observation", "filter"]
    pixel_scale_sw: float = 0.025
    pixel_scale_lw: float = 0.05
    tile_size: int = 4096
    tile_overlap: int = 128
    min_coverage: float = 0.02
    obs_epoch: float = 2023.6
    star_fwhm_px: float = 2.2
    pixfrac: float = 0.8
    in_memory: bool = False
    work_dir: str = "stage3_work"
    clean_work: bool = False


class Recipe(BaseModel):
    """A parsed ``reduce`` recipe: a pipeline, a selection and per-step controls."""

    pipeline: Literal["clear", "grism"] = "clear"
    select: Selection
    steps: dict[str, StepConfig] = {}
    output_noobox: str | None = None
    mute_jwst: bool = False
    grism: GrismOptions = GrismOptions()
    stage3: Stage3Options = Stage3Options()

    @model_validator(mode="after")
    def _known_steps(self) -> "Recipe":
        from noobfriend.cli.reduce._registry import step_names_for

        valid = step_names_for(self.pipeline, self.select.stage)
        unknown = set(self.steps) - valid
        if unknown:
            raise ValueError(
                f"unknown step(s) {sorted(unknown)} for pipeline {self.pipeline!r}; "
                f"valid steps are {sorted(valid)}."
            )
        return self


def ordered(
    chain: tuple[StepSpec, ...], steps: dict[str, StepConfig]
) -> list[tuple[StepSpec, StepConfig]]:
    """Return ``(spec, config)`` in chain order, excluding skipped steps."""
    return [
        (spec, steps.get(spec.name, StepConfig()))
        for spec in chain
        if not steps.get(spec.name, StepConfig()).skip
    ]


def load_recipe(path: Path) -> Recipe:
    """Parse a recipe TOML file into a validated :class:`Recipe`."""
    return Recipe.model_validate(tomllib.loads(Path(path).read_text()))
