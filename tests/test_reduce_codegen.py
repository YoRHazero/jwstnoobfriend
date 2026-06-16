"""Tests for the ``reduce`` recipe model, stage validation and codegen.

Covers the shared :class:`Recipe`, the CLEAR (per-frame) renderer and the GRISM
(two-pass, per-module template) renderer, plus env-based stage validation.
"""

import ast
import tomllib

import pytest

from noobfriend.cli.reduce._codegen_clear_stage2 import render as render_clear
from noobfriend.cli.reduce._codegen_grism_stage2 import render as render_grism
from noobfriend.cli.reduce._io import validate_stages
from noobfriend.cli.reduce._recipe import Recipe, StepConfig
from noobfriend.cli.reduce._reduce_clear_stage2 import scaffold as scaffold_clear
from noobfriend.cli.reduce._reduce_grism_stage2 import scaffold as scaffold_grism


def _clear(**select: object) -> Recipe:
    recipe = Recipe.model_validate(tomllib.loads(scaffold_clear("2a")))
    for field, value in select.items():
        setattr(recipe.select, field, value)
    return recipe


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


# -- GRISM (WFSS) renderer ----------------------------------------------------


def test_grism_render_is_valid_two_pass_python() -> None:
    source = render_grism(_grism())
    ast.parse(source)
    # pass 1: per-frame chain incl. the WFSS master-sky direct call
    assert "AssignWcsStep.call(model)" in source
    assert "subtract_wfss_bkg(" in source
    assert "FlatFieldStep.call(model)" in source
    assert "grism_trace_mask(model.data, model.dq)" in source
    # template build + pass 2
    assert "combine_sky_template(" in source
    assert "sky_residual_grid(model.data" in source
    assert "subtract_sky_template(" in source
    assert "DO_TEMPLATE = True" in source
    assert "SCALAR = True" in source
    assert "DOWNSAMPLE = 4" in source


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


def test_grism_skip_template_disables_second_pass() -> None:
    recipe = _grism()
    recipe.steps["template_bkg"] = StepConfig(skip=True)
    source = render_grism(recipe)
    # the two-pass scaffold stays, gated off by the runtime flag (still valid Python)
    assert "DO_TEMPLATE = False" in source
    ast.parse(source)


# -- recipe / stage validation ------------------------------------------------


def test_unknown_step_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown step"):
        Recipe.model_validate({"select": {"stage": "2a"}, "steps": {"bogus": {}}})


def test_grism_recipe_accepts_grism_steps() -> None:
    recipe = _grism()
    assert recipe.pipeline == "grism"
    assert recipe.grism.downsample == 4


def test_validate_stages_requires_a_save_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAGE_2A_PATH", "/data/2a")
    recipe = _clear()
    for cfg in recipe.steps.values():
        cfg.save_as = None
    with pytest.raises(ValueError, match="no save_as"):
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


def test_validate_stages_grism_needs_2b_and_2bii(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for stage in ("2A", "2B", "2BII"):
        monkeypatch.setenv(f"STAGE_{stage}_PATH", f"/data/{stage.lower()}")
    validate_stages(_grism())  # does not raise
