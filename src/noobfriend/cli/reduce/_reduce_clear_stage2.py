"""The CLEAR (NIRCam imaging) stage-2 reduction chain and its starter recipe.

The fixed, ordered rate -> cal chain (with an optional 2-D background save-point)
and the per-step name -> code mapping. The renderer for this chain lives in
:mod:`noobfriend.cli.reduce._codegen_clear_stage2`.
"""

from noobfriend.cli.reduce._recipe import StepSpec

#: Input stage this chain reduces from.
INPUT_STAGE: str = "2a"

#: The fixed, ordered NIRCam imaging chain (rate -> cal).
#:
#: Flat is applied *before* the 1/f step: the pixel-to-pixel responsivity
#: (crosshatch) is multiplicative in the sky, so flat-fielding only cancels it
#: while the sky is still present -- a 1/f step that removes the sky pedestal
#: first would strand the crosshatch. Outlier flagging follows flat (cleaner
#: detection once the crosshatch is gone) and is DQ-only, so its position is free.
CHAIN: tuple[StepSpec, ...] = (
    StepSpec("jwst_assign_wcs", "WCS assignment", jwst="jwst.assign_wcs.AssignWcsStep"),
    StepSpec("jwst_flat_field", "flat fielding", jwst="jwst.flatfield.FlatFieldStep"),
    StepSpec(
        "jwst_bkg_subtract",
        "official BackgroundStep (no-op without dedicated background exposures)",
        jwst="jwst.background.BackgroundStep",
    ),
    StepSpec(
        "outlier_flag",
        "hot / cosmic-ray outlier-pixel flagging (custom)",
        custom="flag_outlier_pixels",
    ),
    StepSpec("oneoverf", "1/f striping removal (custom)", custom="subtract_oneoverf"),
    StepSpec("jwst_photom", "flux calibration", jwst="jwst.photom.PhotomStep"),
    StepSpec(
        "background_2d",
        "2-D self-background subtraction (custom)",
        custom="subtract_background",
    ),
)

#: Default ``save_as`` save-points for the starter recipe.
SAVE_POINTS: dict[str, str] = {"jwst_photom": "2b", "background_2d": "2bi"}

#: Valid step names for this chain (recipe validation).
STEP_NAMES: frozenset[str] = frozenset(spec.name for spec in CHAIN)


def scaffold(stage: str = INPUT_STAGE) -> str:
    """Return a starter CLEAR recipe listing every step in order, none skipped."""
    lines = [
        "# noobfriend CLEAR (imaging) stage-2 reduction recipe.",
        "# [select] picks which NooBox files to reduce; each step below carries",
        '# skip (default false) and an optional save_as = "<stage>" save-point.',
        "# Step parameters live in the generated code, not here.",
        "",
        'pipeline = "clear"',
        "",
        '# output_noobox = "data/reduced_noobox.json"  '
        "# omit to update NOOBOX_PATH in place",
        "# mute_jwst = true  # silence jwst / CRDS / stpipe INFO+WARNING logging",
        "# workers = 4  # parallel frame processes (single-writer manifest);"
        " 1 = sequential",
        "",
        "[select]",
        f'stage = "{stage}"',
        'pupil = "CLEAR"',
        '# filter = ["F182M", "F210M"]',
        '# detector = "nrca*"',
        "",
    ]
    for spec in CHAIN:
        lines += [f"# {spec.title}", f"[steps.{spec.name}]", "skip = false"]
        if spec.name in SAVE_POINTS:
            lines.append(f'save_as = "{SAVE_POINTS[spec.name]}"')
        lines.append("")
    return "\n".join(lines)
