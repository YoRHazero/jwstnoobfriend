"""Recipe model and the fixed step registry for the ``reduce`` codegen.

A recipe declares two things: which NooBox files to reduce (``[select]`` --
stage / pupil / filter / detector) and, per pipeline step, whether to skip it
(``skip``) and whether to save the frame after it as a given stage (``save_as``).

The ordered set of steps and the step-name -> code mapping live *here* (in the
CLI, not in the ``reduction`` library, which carries no recipe/registry). A
recipe never names a Python function or a jwst class and never carries step
parameters: those are rendered into the generated script with the functions' own
defaults and edited there.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, model_validator

_Criterion = str | list[str] | None


@dataclass(frozen=True)
class StepSpec:
    """One pipeline step: its recipe key and the code it maps to.

    Exactly one of ``custom`` / ``jwst`` is set: ``custom`` names a
    :mod:`noobfriend.reduction` function (called on ``(data, err, dq)``); ``jwst``
    is a dotted ``jwst`` ``Step`` class path (called via ``.call(model)``).
    """

    name: str
    title: str
    custom: str | None = None
    jwst: str | None = None


#: The fixed, ordered NIRCam imaging chain (rate -> cal, with an optional 2-D
#: background save-point). The CLI owns this order and the name -> code mapping.
IMAGING_CHAIN: tuple[StepSpec, ...] = (
    StepSpec("oneoverf", "1/f striping removal (custom)", custom="subtract_oneoverf"),
    StepSpec(
        "outlier_flag",
        "hot / cosmic-ray outlier-pixel flagging (custom)",
        custom="flag_outlier_pixels",
    ),
    StepSpec(
        "jwst_bkg_subtract",
        "official BackgroundStep (no-op without dedicated background exposures)",
        jwst="jwst.background.BackgroundStep",
    ),
    StepSpec("jwst_assign_wcs", "WCS assignment", jwst="jwst.assign_wcs.AssignWcsStep"),
    StepSpec("jwst_flat_field", "flat fielding", jwst="jwst.flatfield.FlatFieldStep"),
    StepSpec("jwst_photom", "flux calibration", jwst="jwst.photom.PhotomStep"),
    StepSpec(
        "background_2d",
        "2-D self-background subtraction (custom)",
        custom="subtract_background",
    ),
)

_STEP_NAMES: frozenset[str] = frozenset(spec.name for spec in IMAGING_CHAIN)


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


class Recipe(BaseModel):
    """A parsed ``reduce`` recipe: a selection plus per-step skip / save-points."""

    select: Selection
    steps: dict[str, StepConfig] = {}
    output_noobox: str | None = None
    mute_jwst: bool = False

    @model_validator(mode="after")
    def _known_steps(self) -> "Recipe":
        unknown = set(self.steps) - _STEP_NAMES
        if unknown:
            raise ValueError(
                f"unknown step(s) {sorted(unknown)}; valid steps are "
                f"{[spec.name for spec in IMAGING_CHAIN]}."
            )
        return self

    def ordered(self) -> list[tuple[StepSpec, StepConfig]]:
        """Return ``(spec, config)`` in canonical order, excluding skipped steps."""
        return [
            (spec, self.steps.get(spec.name, StepConfig()))
            for spec in IMAGING_CHAIN
            if not self.steps.get(spec.name, StepConfig()).skip
        ]


def load_recipe(path: Path) -> Recipe:
    """Parse a recipe TOML file into a validated :class:`Recipe`."""
    return Recipe.model_validate(tomllib.loads(Path(path).read_text()))


def scaffold(stage: str = "2a") -> str:
    """Return a starter recipe TOML listing every step in order, none skipped."""
    lines = [
        "# noobfriend reduce recipe.",
        "# [select] picks which NooBox files to reduce; each step below carries",
        '# skip (default false) and an optional save_as = "<stage>" save-point.',
        "# Step parameters live in the generated code, not here.",
        "",
        '# output_noobox = "data/reduced_noobox.json"  '
        "# omit to update NOOBOX_PATH in place",
        "# mute_jwst = true  # silence jwst / CRDS / stpipe INFO+WARNING logging",
        "",
        "[select]",
        f'stage = "{stage}"',
        '# pupil = "CLEAR"',
        '# filter = ["F182M", "F210M"]',
        '# detector = "nrca*"',
        "",
    ]
    for spec in IMAGING_CHAIN:
        save_as = {"jwst_photom": "2b", "background_2d": "2bi"}.get(spec.name)
        lines += [f"# {spec.title}", f"[steps.{spec.name}]", "skip = false"]
        if save_as is not None:
            lines.append(f'save_as = "{save_as}"')
        lines.append("")
    return "\n".join(lines)
