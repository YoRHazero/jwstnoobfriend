"""Tests for the ``box`` CLI: init / show commands, helpers, and presenters.

The CLI runs in *thin* mode against empty ``.fits`` files (no real FITS is
opened), and the settings-backed helpers are exercised with a stubbed
``get_settings`` so the tests never depend on a project ``.env``.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from noobfriend.cli.box._io import resolve_box_path
from noobfriend.cli.box._options import resolve_stage
from noobfriend.cli.box._presenters import (
    make_box_listing_table,
    make_box_summary_table,
)
from noobfriend.cli.box.app import box_app
from noobfriend.navigation import NooBook, NooBox

runner = CliRunner()


def _touch_fits(directory: Path, name: str) -> None:
    (directory / name).write_text("")


def _thin_box(*names_stages: tuple[str, str]) -> NooBox:
    box = NooBox()
    for name, stage in names_stages:
        box.add(NooBook.from_name(f"/data/{name}", stage))
    return box


# -- helpers ------------------------------------------------------------------


def test_resolve_stage_prefers_argument():
    assert resolve_stage("2bii") == "2bii"


def test_resolve_stage_falls_back_to_start_stage(monkeypatch):
    monkeypatch.setattr(
        "noobfriend.cli.box._options.get_settings",
        lambda: SimpleNamespace(start_stage="1b"),
    )
    assert resolve_stage(None) == "1b"


def test_resolve_stage_raises_when_unset(monkeypatch):
    monkeypatch.setattr(
        "noobfriend.cli.box._options.get_settings",
        lambda: SimpleNamespace(start_stage=None),
    )
    with pytest.raises(typer.BadParameter, match="START_STAGE"):
        resolve_stage(None)


def test_resolve_box_path_prefers_argument(tmp_path):
    target = tmp_path / "noobox.json"
    assert resolve_box_path(target) == target


def test_resolve_box_path_raises_when_unset(monkeypatch):
    monkeypatch.setattr(
        "noobfriend.cli.box._io.get_settings",
        lambda: SimpleNamespace(noobox_path=None),
    )
    with pytest.raises(typer.BadParameter, match="NOOBOX_PATH"):
        resolve_box_path(None)


# -- presenters ---------------------------------------------------------------


def test_summary_table_rows_and_caption():
    box = _thin_box(
        ("jw01345001001_02201_00001_nrca1_rate.fits", "1b"),
        ("jw01345001001_02201_00001_nrca1_cal.fits", "2a"),
        ("jw01345001001_02201_00002_nrca1_cal.fits", "2a"),
    )
    table = make_box_summary_table(Path("noobox.json"), box)

    assert table.row_count == 2  # one row per distinct stage
    assert "noobox.json" in str(table.title)
    assert "3 books" in table.caption


def test_listing_table_one_row_per_product():
    box = _thin_box(
        ("jw01345001001_02201_00001_nrca1_cal.fits", "2a"),
        ("jw01345001001_02201_00001_nrcb1_cal.fits", "2a"),
    )
    table = make_box_listing_table(list(box))
    assert table.row_count == 2


# -- box init -----------------------------------------------------------------


def test_init_writes_single_stage_manifest(tmp_path):
    stage_dir = tmp_path / "stage1b"
    stage_dir.mkdir()
    _touch_fits(stage_dir, "jw01345001001_02201_00001_nrca1_rate.fits")
    _touch_fits(stage_dir, "jw01345001001_02201_00002_nrca1_rate.fits")
    manifest = tmp_path / "noobox.json"

    result = runner.invoke(
        box_app, ["init", "1b", "--root", str(stage_dir), "--path", str(manifest)]
    )

    assert result.exit_code == 0
    assert manifest.exists()
    box = NooBox.load(manifest)
    assert len(box) == 2
    assert all(book.stage == "1b" for book in box)


def test_init_empty_match_writes_nothing(tmp_path):
    stage_dir = tmp_path / "empty"
    stage_dir.mkdir()
    manifest = tmp_path / "noobox.json"

    result = runner.invoke(
        box_app, ["init", "1b", "--root", str(stage_dir), "--path", str(manifest)]
    )

    assert result.exit_code == 1
    assert not manifest.exists()


def test_init_refuses_to_clobber_without_force(tmp_path):
    stage_dir = tmp_path / "stage1b"
    stage_dir.mkdir()
    _touch_fits(stage_dir, "jw01345001001_02201_00001_nrca1_rate.fits")
    manifest = tmp_path / "noobox.json"
    manifest.write_text("SENTINEL")

    result = runner.invoke(
        box_app,
        ["init", "1b", "--root", str(stage_dir), "--path", str(manifest)],
        input="n\n",
    )

    assert result.exit_code != 0
    assert manifest.read_text() == "SENTINEL"  # untouched


# -- box show -----------------------------------------------------------------


def test_show_reads_manifest(tmp_path):
    manifest = tmp_path / "noobox.json"
    _thin_box(
        ("jw01345001001_02201_00001_nrca1_rate.fits", "1b"),
        ("jw01345001001_02201_00001_nrca1_cal.fits", "2a"),
    ).save(manifest)

    result = runner.invoke(box_app, ["show", "--path", str(manifest)])

    assert result.exit_code == 0


def test_show_stage_filter(tmp_path):
    manifest = tmp_path / "noobox.json"
    _thin_box(
        ("jw01345001001_02201_00001_nrca1_rate.fits", "1b"),
        ("jw01345001001_02201_00001_nrca1_cal.fits", "2a"),
    ).save(manifest)

    result = runner.invoke(box_app, ["show", "--path", str(manifest), "--stage", "2a"])

    assert result.exit_code == 0


def test_show_missing_manifest_errors(tmp_path):
    result = runner.invoke(box_app, ["show", "--path", str(tmp_path / "absent.json")])
    assert result.exit_code != 0
