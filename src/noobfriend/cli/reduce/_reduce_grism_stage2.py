"""The GRISM (NIRCam WFSS) stage-2 reduction chain and its starter recipe.

Unlike the per-frame CLEAR chain, this pipeline is two-pass: pass 1 reduces each
frame rate -> 2b (flag_outlier -> assign_wcs -> WFSS master-sky -> flat) and
collects a downsampled trace-masked residual; between passes a per-(module,
direction) sky-residual template is built from those residuals; pass 2 subtracts
the template (2b -> 2bii). The master-sky and template-background steps are not
plain ``Step.call`` / ``(data, err, dq)`` ops, so the renderer in
:mod:`noobfriend.cli.reduce._codegen_grism_stage2` emits their code directly --
the ``StepSpec`` entries here carry only names, titles and skip/save semantics.
"""

from noobfriend.cli.reduce._recipe import StepSpec

#: Input stage this chain reduces from.
INPUT_STAGE: str = "2a"

#: The fixed, ordered WFSS chain. ``master_sky`` / ``template_bkg`` are rendered
#: bespoke (no ``custom`` / ``jwst`` mapping); the rest are ordinary ops.
CHAIN: tuple[StepSpec, ...] = (
    StepSpec(
        "flag_outlier",
        "hot / cosmic-ray outlier-pixel flagging (custom)",
        custom="flag_outlier_pixels",
    ),
    StepSpec(
        "assign_wcs", "grism WCS assignment", jwst="jwst.assign_wcs.AssignWcsStep"
    ),
    StepSpec("master_sky", "WFSS master-sky subtraction (per-module WFSSBKG)"),
    StepSpec("flat", "flat fielding", jwst="jwst.flatfield.FlatFieldStep"),
    StepSpec(
        "template_bkg", "per-(module,direction) sky-residual template subtraction"
    ),
)

#: Default ``save_as`` save-points for the starter recipe.
SAVE_POINTS: dict[str, str] = {"flat": "2b", "template_bkg": "2bii"}

#: Valid step names for this chain (recipe validation).
STEP_NAMES: frozenset[str] = frozenset(spec.name for spec in CHAIN)


def scaffold(stage: str = INPUT_STAGE) -> str:
    """Return a starter GRISM recipe listing every step in order, none skipped."""
    lines = [
        "# noobfriend GRISM (NIRCam WFSS) stage-2 reduction recipe.",
        "# Two-pass: rate -> 2b (per frame), build per-module sky template, 2b -> 2bii.",
        "# Step parameters live in the generated code, not here.",
        "",
        'pipeline = "grism"',
        "",
        '# output_noobox = "data/reduced_noobox.json"  '
        "# omit to update NOOBOX_PATH in place",
        "# mute_jwst = true  # silence jwst / CRDS / stpipe INFO+WARNING logging",
        "",
        "[select]",
        f'stage = "{stage}"',
        'pupil = ["GRISMR", "GRISMC"]  # dispersed LW frames (the grism images)',
        '# detector = "nrc?long"',
        "",
        "# Per-(module, direction) sky-residual template options.",
        "[grism]",
        "scalar = true       # per-frame scalar fit of the template (recommended)",
        "downsample = 4      # residual-grid downsample factor (512^2 grid at 4)",
        "smooth_sigma = 24.0 # Gaussian smoothing of the upsampled template",
        "",
    ]
    for spec in CHAIN:
        lines += [f"# {spec.title}", f"[steps.{spec.name}]", "skip = false"]
        if spec.name in SAVE_POINTS:
            lines.append(f'save_as = "{SAVE_POINTS[spec.name]}"')
        lines.append("")
    return "\n".join(lines)
