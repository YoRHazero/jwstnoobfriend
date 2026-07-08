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
    pipeline's own renderer emits its code. ``skippable`` / ``saveable`` define
    which generic recipe controls the renderer actually supports.
    """

    name: str
    title: str
    custom: str | None = None
    jwst: str | None = None
    skippable: bool = True
    saveable: bool = True


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
    """Grism-pipeline knobs for the 2b -> 2bii sky-residual template step.

    ``template_dir`` stores the per-(module, direction) sky-residual templates
    (``skytemplate_modA_R.fits`` etc.).
    """

    template_dir: str = "grism_templates"
    scalar: bool = True
    downsample: int = 4
    smooth_sigma: float = 24.0


class Stage3Options(BaseModel):
    """Stage-3 mosaicking knobs: grouping, unified output grid and astrometry.

    ``engine`` picks the whole stage-3 implementation, with no mixing:
    ``"noob"`` (default) runs the noobase-backed custom chain (deep-catalog
    align -> scalar sky match -> field-median outlier -> reproject_exact coadd,
    writing an asdf-in-FITS 3a with an honest error plane and a noise kernel);
    ``"jwst"`` runs the official ``TweakRegStep`` / ``SkyMatchStep`` /
    ``OutlierDetectionStep`` / ``ResampleStep`` chain (the ``outlier_engine`` /
    ``in_memory`` / ``minobj`` knobs below apply only to it). ``group_by`` are
    the :class:`~noobfriend.navigation.NooBook` fields whose
    distinct combinations define one mosaic (``["observation", "filter"]`` =
    one per field per band). The output scale is per channel --
    ``pixel_scale_sw`` for short-wave detectors (``nrc[ab][1-4]``) and
    ``pixel_scale_lw`` for long-wave (``nrc[ab]long``); set them equal for one
    unified grid. ``align_to_roll`` (default) rolls the output grid to the
    exposures' position angle so a single-roll field fills a tight rectangle
    instead of a diagonal band in a north-up box (set false to force north-up).
    Each mosaic is tiled into evenly-sized ``tile_size`` cells with
    a ``tile_overlap`` border, and tiles whose real footprint coverage is below
    ``min_coverage`` are dropped. ``obs_epoch`` is the decimal-year target for
    GAIA proper-motion propagation and ``star_fwhm_px`` the expected
    point-source FWHM for image-source detection. ``minobj`` / ``abs_minobj`` are
    the tweakreg relative / absolute minimum-match thresholds: the absolute tie
    is matched image-source-to-GAIA, and the jwst default (15) silently rejects
    the fit -- leaving the WCS un-corrected -- when fewer clean sources coincide
    with GAIA, so the default here is lowered (verified: an F444W field tied at
    ~6 mas with ~12 matches). ``outlier_engine`` picks the cross-exposure
    CR flagger: ``"noob"`` (default) is the noobase-backed field-median engine
    (streaming, memory-bounded, no hidden full-field drizzle), ``"jwst"`` falls
    back to ``OutlierDetectionStep``. ``in_memory`` controls whether the prep steps
    hold the group's models resident: ``"auto"`` (default) keeps a group in
    memory only when its estimated peak (frames x planes x a resampling
    headroom) fits half the machine's RAM, spilling to disk otherwise; ``true``
    / ``false`` force one mode for every group. Per-group astrometry sidecars
    go under ``work_dir`` and are deleted afterwards when ``clean_work`` is
    set.
    """

    engine: Literal["noob", "jwst"] = "noob"
    group_by: list[str] = ["observation", "filter"]
    pixel_scale_sw: float = 0.025
    pixel_scale_lw: float = 0.05
    align_to_roll: bool = True
    tile_size: int = 4096
    tile_overlap: int = 128
    min_coverage: float = 0.02
    obs_epoch: float = 2023.6
    star_fwhm_px: float = 2.2
    minobj: int = 7
    abs_minobj: int = 6
    pixfrac: float = 0.8
    outlier_engine: Literal["noob", "jwst"] = "noob"
    in_memory: bool | Literal["auto"] = "auto"
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
        from noobfriend.cli.reduce._registry import step_specs_for

        specs = step_specs_for(self.pipeline, self.select.stage)
        if not specs:
            raise ValueError(
                f"no reduction pipeline for pipeline={self.pipeline!r} and "
                f"select.stage={self.select.stage!r}."
            )
        unknown = set(self.steps) - set(specs)
        if unknown:
            raise ValueError(
                f"unknown step(s) {sorted(unknown)} for pipeline {self.pipeline!r}; "
                f"valid steps are {sorted(specs)}."
            )
        for name, cfg in self.steps.items():
            spec = specs[name]
            if cfg.skip and not spec.skippable:
                raise ValueError(f"step {name!r} cannot be skipped in this pipeline.")
            if cfg.save_as is not None and not spec.saveable:
                raise ValueError(
                    f"step {name!r} cannot define save_as in this pipeline."
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
