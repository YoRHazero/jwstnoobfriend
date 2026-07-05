"""The CLEAR (NIRCam imaging) stage-3 mosaicking chain and its starter recipe.

Unlike the per-frame stage-2 chains, stage-3 is *group-based*: 2bi cal frames are
partitioned into mosaics by metadata (``[stage3].group_by``, e.g. field + band),
and each group runs a field-level prep (align -> skymatch -> outlier) before
being drizzled onto a unified, tiled output grid (resample -> 3a). None of the
four steps is a plain ``Step.call`` / ``(data, err, dq)`` op -- ``align`` feeds
``TweakRegStep`` cleaned image catalogs and a PM-propagated GAIA ``abs_refcat``,
and ``resample`` runs per output tile -- so the renderer in
:mod:`noobfriend.cli.reduce._codegen_clear_stage3` emits their code directly; the
``StepSpec`` entries here carry only names, titles and skip/save semantics.
"""

from noobfriend.cli.reduce._recipe import StepSpec

#: Input stage this chain mosaics from.
INPUT_STAGE: str = "2bi"

#: The fixed, ordered stage-3 chain. Every step is rendered bespoke (no
#: ``custom`` / ``jwst`` mapping); ``resample`` carries the 3a save-point.
CHAIN: tuple[StepSpec, ...] = (
    StepSpec(
        "align",
        "absolute astrometric alignment (tweakreg + cleaned image catalogs & GAIA)",
        saveable=False,
    ),
    StepSpec("skymatch", "cross-exposure background matching", saveable=False),
    StepSpec("outlier", "cross-exposure outlier / cosmic-ray flagging", saveable=False),
    StepSpec(
        "resample", "drizzle onto the unified, tiled output grid", skippable=False
    ),
)

#: Default ``save_as`` save-points for the starter recipe.
SAVE_POINTS: dict[str, str] = {"resample": "3a"}

#: Valid step names for this chain (recipe validation).
STEP_NAMES: frozenset[str] = frozenset(spec.name for spec in CHAIN)


def scaffold(stage: str = INPUT_STAGE) -> str:
    """Return a starter CLEAR stage-3 recipe listing every step in order."""
    lines = [
        "# noobfriend CLEAR (imaging) stage-3 mosaicking recipe.",
        "# 2bi cal frames are grouped into mosaics (one per [stage3].group_by combo),",
        "# field-level aligned / sky-matched / outlier-flagged, then drizzled onto a",
        "# unified, tiled output grid. Step parameters live in the generated code.",
        "",
        'pipeline = "clear"',
        "",
        '# output_noobox = "data/mosaic_noobox.json"  '
        "# omit to update NOOBOX_PATH in place",
        "# mute_jwst = true  # silence jwst / CRDS / stpipe INFO+WARNING logging",
        "",
        "[select]",
        f'stage = "{stage}"',
        'pupil = "CLEAR"',
        '# filter = ["F182M", "F210M", "F444W"]',
        "",
        "# Mosaic grouping, unified output grid and astrometry options.",
        "[stage3]",
        'group_by = ["observation", "filter"]  # one mosaic per field per band',
        "pixel_scale_sw = 0.025  # arcsec/pix for short-wave (nrc[ab][1-4])",
        "pixel_scale_lw = 0.05   # arcsec/pix for long-wave (nrc[ab]long)",
        "#   set the two equal for a single unified output grid across all bands",
        "align_to_roll = true # roll the grid to the exposures (false = north-up)",
        "tile_size = 4096     # evenly-sized output tile size (px)",
        "tile_overlap = 128   # tile overlap border (px)",
        "min_coverage = 0.02  # drop tiles below this footprint coverage fraction",
        "obs_epoch = 2023.6   # decimal year for GAIA proper-motion propagation",
        "star_fwhm_px = 2.2   # expected point-source FWHM for source detection",
        "minobj = 7           # tweakreg relative min matches per group",
        "abs_minobj = 6       # tweakreg absolute (GAIA) min matches; jwst default 15",
        "#   raising abs_minobj too high silently leaves the WCS un-tied to GAIA",
        "pixfrac = 0.8        # drizzle pixfrac",
        'outlier_engine = "noob"  # field-median CR flagger; "jwst" = OutlierDetectionStep',
        'in_memory = "auto"   # keep a group in RAM when it fits; true/false to force',
        'work_dir = "stage3_work"  # scratch dir for per-group tweakreg catalogs/refcat',
        "clean_work = false   # delete each group's work_dir sidecars after it finishes",
        "",
    ]
    for spec in CHAIN:
        lines += [f"# {spec.title}", f"[steps.{spec.name}]"]
        if spec.skippable:
            lines.append("skip = false")
        if spec.name in SAVE_POINTS:
            lines.append(f'save_as = "{SAVE_POINTS[spec.name]}"')
        lines.append("")
    return "\n".join(lines)
