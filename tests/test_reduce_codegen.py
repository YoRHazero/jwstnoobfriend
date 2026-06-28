"""Tests for the ``reduce`` recipe model, stage validation and codegen.

Covers the shared :class:`Recipe`, the CLEAR (per-frame) renderer and the GRISM
(two-pass, per-module template) renderer, plus env-based stage validation.
"""

import ast
import tomllib
from pathlib import Path

import pytest

from noobfriend.cli.reduce._codegen_clear_stage2 import render as render_clear
from noobfriend.cli.reduce._codegen_clear_stage3 import render as render_clear3
from noobfriend.cli.reduce._codegen_grism_stage2 import render as render_grism
from noobfriend.cli.reduce.gen import _ensure_recipe_matches_pipeline
from noobfriend.cli.reduce._io import validate_stages
from noobfriend.cli.reduce._recipe import Recipe, StepConfig
from noobfriend.cli.reduce._reduce_clear_stage2 import scaffold as scaffold_clear
from noobfriend.cli.reduce._reduce_clear_stage3 import scaffold as scaffold_clear3
from noobfriend.cli.reduce._reduce_grism_stage2 import scaffold as scaffold_grism
from noobfriend.cli.reduce._registry import select_pipelines


def _clear(**select: object) -> Recipe:
    recipe = Recipe.model_validate(tomllib.loads(scaffold_clear("2a")))
    for field, value in select.items():
        setattr(recipe.select, field, value)
    return recipe


def _clear3() -> Recipe:
    return Recipe.model_validate(tomllib.loads(scaffold_clear3("2bi")))


def _grism() -> Recipe:
    return Recipe.model_validate(tomllib.loads(scaffold_grism("2a")))


# -- CLEAR (imaging) renderer -------------------------------------------------


def test_clear_render_is_valid_python_with_the_chain() -> None:
    source = render_clear(_clear(pupil="CLEAR", filter=["F182M", "F210M"]))
    ast.parse(source)
    assert "subtract_oneoverf(model.data, model.err, model.dq)" in source
    assert "flag_outlier_pixels(model.data, model.err, model.dq)" in source
    assert "PhotomStep.call(model)" in source
    assert "BackgroundStep.call(model)" in source  # jwst no-op is included
    assert "from noobfriend.reduction import" in source
    assert "SELECT = {'pupil': 'CLEAR', 'filter': ['F182M', 'F210M']}" in source


def test_clear_save_points_thread_explicit_lineage() -> None:
    source = render_clear(_clear())
    assert "write_bytes(_loc, _raw)" in source
    assert "NooBook.from_file(_loc, '2b', parents=[parent], raw=_raw)" in source
    assert "NooBook.from_file(_loc, '2bi', parents=[parent], raw=_raw)" in source
    assert source.index("'2b'") < source.index("'2bi'")


def test_clear_skip_removes_a_custom_step() -> None:
    recipe = _clear()
    recipe.steps["oneoverf"] = StepConfig(skip=True)
    source = render_clear(recipe)
    assert "subtract_oneoverf" not in source
    assert "flag_outlier_pixels" in source


def test_mute_jwst_renders_logging_disable() -> None:
    recipe = _clear()
    recipe.mute_jwst = True
    assert "logging.disable(logging.WARNING)" in render_clear(recipe)
    assert "logging.disable" not in render_clear(_clear())


def test_clear_fixed_pattern_runs_two_pass_before_oneoverf() -> None:
    source = render_clear(_clear())
    ast.parse(source)
    # pass 0: per-detector template build helper + call
    assert "def _build_fixed_pattern_templates(box)" in source
    assert "templates = _build_fixed_pattern_templates(box)" in source
    assert "fixed_pattern_residual(model.data, model.dq)" in source
    assert "combine_fixed_pattern(" in source
    assert "FIXED_PATTERN_DIR = Path('clear_templates')" in source
    # per-frame subtract uses the detector's template, and runs before 1/f
    assert (
        "subtract_fixed_pattern(model.data, model.err, model.dq, templates[book.detector])"
        in source
    )
    assert source.index("subtract_fixed_pattern(") < source.index("subtract_oneoverf(")


def test_clear_fixed_pattern_template_dir_is_configurable() -> None:
    recipe = _clear()
    recipe.clear.template_dir = "scratch/fp"
    assert "FIXED_PATTERN_DIR = Path('scratch/fp')" in render_clear(recipe)


def test_clear_skip_fixed_pattern_omits_the_pass() -> None:
    recipe = _clear()
    recipe.steps["fixed_pattern"] = StepConfig(skip=True)
    source = render_clear(recipe)
    ast.parse(source)
    assert "subtract_fixed_pattern" not in source
    assert "_build_fixed_pattern_templates" not in source
    assert "FIXED_PATTERN_DIR" not in source
    assert "subtract_oneoverf(" in source  # the rest of the chain still renders


# -- GRISM (WFSS) renderer ----------------------------------------------------


def test_grism_render_is_valid_two_pass_python() -> None:
    source = render_grism(_grism())
    ast.parse(source)
    # pass 0: rate-stage fixed-pattern template build + cache
    assert "DO_FIXED_PATTERN = True" in source
    assert "def _build_fixed_pattern_templates(box)" in source
    assert "fixed_templates = _build_fixed_pattern_templates(box)" in source
    assert "fixedpattern_mod{module}_{direction}.npy" in source
    assert "grism_fixed_pattern_frame(" in source
    assert "combine_fixed_pattern(" in source
    assert "TEMPLATE_DIR = Path('grism_templates')" in source
    # fixed-pattern subtraction is the first pass-1 operation
    assert "subtract_grism_fixed_pattern(" in source
    assert source.index("subtract_grism_fixed_pattern(") < source.index(
        "flag_outlier_pixels("
    )
    assert source.index("subtract_grism_fixed_pattern(") < source.index(
        "AssignWcsStep.call(model)"
    )
    # pass 1: per-frame chain incl. the WFSS master-sky direct call
    assert "AssignWcsStep.call(model)" in source
    assert "subtract_wfss_bkg(" in source
    assert "FlatFieldStep.call(model)" in source
    assert "def _dispersion_axis(book)" in source
    assert 'pupil == "GRISMR"' in source
    assert 'pupil == "GRISMC"' in source
    assert "Unsupported grism pupil" in source
    assert "dispersion_axis=_dispersion_axis(book)" in source
    assert "dispersion_axis=_dispersion_axis(b2)" in source
    # template build + pass 2
    assert "combine_sky_template(" in source
    assert "sky_residual_grid(model.data" in source
    assert "subtract_sky_template(" in source
    assert "DO_TEMPLATE = True" in source
    assert "SCALAR = True" in source
    assert "DOWNSAMPLE = 4" in source


def test_grism_fixed_pattern_template_dir_is_configurable() -> None:
    recipe = _grism()
    recipe.grism.template_dir = "scratch/grism"
    assert "TEMPLATE_DIR = Path('scratch/grism')" in render_grism(recipe)


def test_grism_skip_fixed_pattern_omits_the_pass() -> None:
    recipe = _grism()
    recipe.steps["fixed_pattern"] = StepConfig(skip=True)
    source = render_grism(recipe)

    ast.parse(source)
    assert "DO_FIXED_PATTERN = False" in source
    assert "subtract_grism_fixed_pattern" not in source
    assert "_build_fixed_pattern_templates" not in source
    assert "grism_fixed_pattern_frame" not in source
    assert "combine_sky_template(" in source
    assert "subtract_sky_template(" in source


def test_grism_threads_2b_then_2bii_lineage() -> None:
    source = render_grism(_grism())
    assert "NooBook.from_file(_loc, STAGE_2B, parents=[parent], raw=_raw)" in source
    assert "NooBook.from_file(_loc, STAGE_2BII, parents=[b2], raw=_raw)" in source
    assert 'STAGE_2B = "2b"' in source
    assert 'STAGE_2BII = "2bii"' in source


def test_grism_skip_master_sky_omits_it() -> None:
    recipe = _grism()
    recipe.steps["master_sky"] = StepConfig(skip=True)
    source = render_grism(recipe)
    assert "subtract_wfss_bkg" not in source
    assert "grism_trace_mask" in source  # still needed for the template


def test_grism_skip_jwst_steps_omits_them() -> None:
    recipe = _grism()
    recipe.steps["assign_wcs"] = StepConfig(skip=True)
    recipe.steps["flat"] = StepConfig(skip=True)

    source = render_grism(recipe)

    assert "AssignWcsStep.call(model)" not in source
    assert "FlatFieldStep.call(model)" not in source
    assert "from jwst.assign_wcs import AssignWcsStep" not in source
    assert "from jwst.flatfield import FlatFieldStep" not in source
    ast.parse(source)


def test_grism_skip_template_disables_second_pass() -> None:
    recipe = _grism()
    recipe.steps["template_bkg"] = StepConfig(skip=True)
    source = render_grism(recipe)
    # the two-pass scaffold stays, gated off by the runtime flag (still valid Python)
    assert "DO_TEMPLATE = False" in source
    ast.parse(source)


# -- CLEAR stage-3 (mosaicking) renderer --------------------------------------


def test_clear3_render_is_valid_group_based_python() -> None:
    source = render_clear3(_clear3())
    ast.parse(source)
    assert "from jwst.tweakreg import TweakRegStep" in source
    assert "TweakRegStep.call(" in source
    assert "SkyMatchStep.call(library, in_memory=IN_MEMORY)" in source
    assert "OutlierDetectionStep.call(library, in_memory=IN_MEMORY)" in source
    assert "ResampleStep.call(" in source
    assert "abs_refcat=str(absref)" in source
    assert "abs_minobj=ABS_MINOBJ" in source and "ABS_MINOBJ = 6" in source
    assert "build_catalog(" in source and "select_point_sources(cat)" in source
    assert "GROUP_BY = ['observation', 'filter']" in source
    assert "PIXEL_SCALE_SW = 0.025" in source and "PIXEL_SCALE_LW = 0.05" in source
    assert "scale = _pixel_scale(gbooks)" in source  # per-channel output scale
    assert 'rotation="auto" if ALIGN_TO_ROLL else 0.0' in source  # roll-aligned grid
    assert 'STAGE_IN = "2bi"' in source and 'STAGE_OUT = "3a"' in source


def test_clear3_uses_modellibrary_borrow_pattern() -> None:
    # stage-3 steps return a ModelLibrary (not the input), so the runner must
    # build a (possibly on-disk) library and borrow/shelve, not iterate a list
    source = render_clear3(_clear3())
    assert "ModelLibrary(_asn(paths), on_disk=not IN_MEMORY)" in source
    assert "library = TweakRegStep.call(" in source
    assert "library.borrow(i) for i in members" in source
    assert "library.shelve(model, index, modify=False)" in source
    # the per-tile subset is in memory, so its resample is always in_memory=True
    assert "in_memory=True" in source


def test_clear3_stamps_many_to_one_tile_lineage() -> None:
    source = render_clear3(_clear3())
    assert "tile_grid(field, target_size=TILE_SIZE, overlap=TILE_OVERLAP)" in source
    assert "tile_members(field, tile, corners)" in source
    assert "parents=[gbooks[i] for i in members]" in source


def test_clear3_skip_align_gates_tweakreg() -> None:
    recipe = _clear3()
    recipe.steps["align"] = StepConfig(skip=True)
    source = render_clear3(recipe)
    # steps are gated by runtime DO_* flags, not by omitting their code/imports
    assert "DO_ALIGN = False" in source
    assert "ResampleStep.call(" in source  # resample still runs
    ast.parse(source)


def test_clear3_recipe_validates_stage3_steps() -> None:
    recipe = _clear3()
    assert recipe.pipeline == "clear"
    assert recipe.select.stage == "2bi"
    assert recipe.stage3.pixel_scale_sw == 0.025
    assert recipe.stage3.pixel_scale_lw == 0.05
    assert set(recipe.steps) == {"align", "skymatch", "outlier", "resample"}


def test_clear3_names_carry_scale_token_and_clean_work_gated() -> None:
    source = render_clear3(_clear3())
    assert "_scale_token(scale)" in source  # scale token in the output filename
    assert "WORK_DIR = Path('stage3_work')" in source
    assert "CLEAN_WORK = False" in source
    assert "shutil.rmtree(work, ignore_errors=True)" in source


# -- recipe / stage validation ------------------------------------------------


def test_unknown_step_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown step"):
        Recipe.model_validate({"select": {"stage": "2a"}, "steps": {"bogus": {}}})


def test_unknown_pipeline_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="no reduction pipeline"):
        Recipe.model_validate({"pipeline": "clear", "select": {"stage": "9z"}})


def test_unsupported_step_save_as_is_rejected() -> None:
    grism = tomllib.loads(scaffold_grism("2a"))
    grism["steps"]["assign_wcs"]["save_as"] = "2wcs"
    with pytest.raises(ValueError, match="cannot define save_as"):
        Recipe.model_validate(grism)

    clear3 = tomllib.loads(scaffold_clear3("2bi"))
    clear3["steps"]["align"]["save_as"] = "3pre"
    with pytest.raises(ValueError, match="cannot define save_as"):
        Recipe.model_validate(clear3)


def test_required_step_skip_is_rejected() -> None:
    data = tomllib.loads(scaffold_clear3("2bi"))
    data["steps"]["resample"]["skip"] = True
    with pytest.raises(ValueError, match="cannot be skipped"):
        Recipe.model_validate(data)


def test_grism_recipe_accepts_grism_steps() -> None:
    recipe = _grism()
    assert recipe.pipeline == "grism"
    assert recipe.grism.downsample == 4
    assert recipe.grism.template_dir == "grism_templates"
    assert "fixed_pattern" in recipe.steps


def test_validate_stages_requires_a_save_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAGE_2A_PATH", "/data/2a")
    recipe = _clear()
    for cfg in recipe.steps.values():
        cfg.save_as = None
    with pytest.raises(ValueError, match="no output stage"):
        validate_stages(recipe)


def test_validate_stages_flags_undefined_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAGE_2A_PATH", "/data/2a")
    monkeypatch.delenv("STAGE_2B_PATH", raising=False)
    monkeypatch.delenv("STAGE_2BI_PATH", raising=False)
    with pytest.raises(ValueError, match="STAGE_<STAGE>_PATH"):
        validate_stages(_clear())


def test_validate_stages_passes_when_all_defined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for stage in ("2A", "2B", "2BI"):
        monkeypatch.setenv(f"STAGE_{stage}_PATH", f"/data/{stage.lower()}")
    validate_stages(_clear())  # does not raise


def test_gen_rejects_recipe_renderer_mismatch() -> None:
    recipe = _grism()
    selected = select_pipelines("clear", "2")[0]

    with pytest.raises(ValueError, match="refusing to render"):
        _ensure_recipe_matches_pipeline(
            Path("recipe_clear_stage2.toml"), selected, recipe
        )


def test_validate_stages_grism_needs_2b_and_2bii(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for stage in ("2A", "2B", "2BII"):
        monkeypatch.setenv(f"STAGE_{stage}_PATH", f"/data/{stage.lower()}")
    validate_stages(_grism())  # does not raise


def test_validate_stages_grism_checks_implicit_pass1_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STAGE_2A_PATH", "/data/2a")
    monkeypatch.delenv("STAGE_2B_PATH", raising=False)
    monkeypatch.setenv("STAGE_2BII_PATH", "/data/2bii")
    recipe = _grism()
    recipe.steps["flat"].skip = True

    with pytest.raises(ValueError, match="STAGE_<STAGE>_PATH"):
        validate_stages(recipe)


def test_validate_stages_grism_accepts_renderer_default_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for stage in ("2A", "2B", "2BII"):
        monkeypatch.setenv(f"STAGE_{stage}_PATH", f"/data/{stage.lower()}")
    recipe = _grism()
    recipe.steps.pop("flat")
    recipe.steps.pop("template_bkg")

    validate_stages(recipe)  # renderer defaults to 2b / 2bii


def test_validate_stages_clear3_accepts_renderer_default_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STAGE_2BI_PATH", "/data/2bi")
    monkeypatch.setenv("STAGE_3A_PATH", "/data/3a")
    recipe = _clear3()
    recipe.steps.pop("resample")

    validate_stages(recipe)  # renderer defaults to 3a
