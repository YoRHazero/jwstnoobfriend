"""Tests for the ``reduce`` recipe model, stage validation and codegen."""

import ast
import tomllib

import pytest

from noobfriend.cli.reduce._codegen import render
from noobfriend.cli.reduce._io import validate_stages
from noobfriend.cli.reduce._recipe import Recipe, StepConfig, scaffold


def _recipe(**select: object) -> Recipe:
    recipe = Recipe.model_validate(tomllib.loads(scaffold("2a")))
    for field, value in select.items():
        setattr(recipe.select, field, value)
    return recipe


def test_render_is_valid_python_with_the_chain() -> None:
    source = render(_recipe(pupil="CLEAR", filter=["F182M", "F210M"]))
    ast.parse(source)  # valid Python
    assert "subtract_oneoverf(model.data, model.err, model.dq)" in source
    assert "flag_outlier_pixels(model.data, model.err, model.dq)" in source
    assert "PhotomStep.call(model)" in source
    assert "BackgroundStep.call(model)" in source  # jwst no-op is included
    assert "from noobfriend.reduction import" in source
    assert "SELECT = {'pupil': 'CLEAR', 'filter': ['F182M', 'F210M']}" in source


def test_save_points_thread_explicit_lineage() -> None:
    source = render(_recipe())
    assert "write_bytes(_loc, _raw)" in source  # local or remote upload
    assert "NooBook.from_file(_loc, '2b', parents=[parent], raw=_raw)" in source
    assert "NooBook.from_file(_loc, '2bi', parents=[parent], raw=_raw)" in source
    # the 2b save-point is rendered before the 2bi one, so 2bi's parent is 2b.
    assert source.index("'2b'") < source.index("'2bi'")


def test_skip_removes_a_custom_step() -> None:
    recipe = _recipe()
    recipe.steps["oneoverf"] = StepConfig(skip=True)
    source = render(recipe)
    assert "subtract_oneoverf" not in source
    assert "flag_outlier_pixels" in source  # the others remain


def test_mute_jwst_renders_logging_disable() -> None:
    recipe = _recipe()
    recipe.mute_jwst = True
    source = render(recipe)
    assert "logging.disable(logging.WARNING)" in source
    # mute is opt-in: the default recipe leaves logging untouched.
    assert "logging.disable" not in render(_recipe())


def test_unknown_step_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown step"):
        Recipe.model_validate({"select": {"stage": "2a"}, "steps": {"bogus": {}}})


def test_validate_stages_requires_a_save_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAGE_2A_PATH", "/data/2a")
    recipe = _recipe()
    for cfg in recipe.steps.values():
        cfg.save_as = None
    with pytest.raises(ValueError, match="no save_as"):
        validate_stages(recipe)


def test_validate_stages_flags_undefined_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STAGE_2A_PATH", "/data/2a")
    monkeypatch.delenv("STAGE_2B_PATH", raising=False)
    monkeypatch.delenv("STAGE_2BI_PATH", raising=False)
    with pytest.raises(ValueError, match="STAGE_<STAGE>_PATH"):
        validate_stages(_recipe())


def test_validate_stages_passes_when_all_defined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for stage in ("2A", "2B", "2BI"):
        monkeypatch.setenv(f"STAGE_{stage}_PATH", f"/data/{stage.lower()}")
    validate_stages(_recipe())  # does not raise
