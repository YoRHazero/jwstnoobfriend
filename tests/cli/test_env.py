"""Unit tests for the pure logic of the env CLI (schema view, render, helpers)."""

from pathlib import Path

import pytest
import typer

from noobfriend.cli.env._io import find_overrides, split_existing
from noobfriend.cli.env._options import start_stage_callback, stage_list_callback
from noobfriend.cli.env._render import render_base
from noobfriend.cli.env.add_stage import stage_dir
from noobfriend.cli.env.check import (
    _group_of,
    _is_remote_path,
    _resolve_server,
    cli_check,
)
from noobfriend.core.env import EnvGroup, EnvPathKind, env_fields


class TestEnvFields:
    """The render-friendly view derived from NoobSettings."""

    def test_view_matches_model(self) -> None:
        fields = {field.name: field for field in env_fields()}
        assert set(fields) == {
            "CRDS_SERVER_URL",
            "CRDS_PATH",
            "START_STAGE",
            "NOOBOX_PATH",
            "NOOB_SERVER",
            "DATA_ROOT_PATH",
        }

    def test_path_fields_flagged(self) -> None:
        fields = {field.name: field for field in env_fields()}
        assert fields["CRDS_PATH"].is_path
        assert fields["DATA_ROOT_PATH"].is_path
        assert fields["NOOBOX_PATH"].is_path
        assert not fields["NOOB_SERVER"].is_path
        assert not fields["CRDS_SERVER_URL"].is_path

    def test_noobox_path_is_a_file_path(self) -> None:
        fields = {field.name: field for field in env_fields()}
        assert fields["NOOBOX_PATH"].path_kind is EnvPathKind.file
        assert fields["DATA_ROOT_PATH"].path_kind is EnvPathKind.directory

    def test_defaults_and_groups(self) -> None:
        fields = {field.name: field for field in env_fields()}
        assert fields["NOOB_SERVER"].default == "localhost"
        assert fields["CRDS_SERVER_URL"].default == "https://jwst-crds.stsci.edu"
        assert fields["START_STAGE"].default is None
        assert fields["CRDS_SERVER_URL"].group is EnvGroup.setup
        assert fields["CRDS_PATH"].group is EnvGroup.setup
        assert fields["NOOBOX_PATH"].group is EnvGroup.noob
        assert fields["NOOB_SERVER"].group is EnvGroup.storage
        assert fields["DATA_ROOT_PATH"].group is EnvGroup.storage


class TestRenderBase:
    """Rendering the static configuration section."""

    def test_unset_variables_are_commented_out(self) -> None:
        rendered = render_base({})
        assert "# START_STAGE=" in rendered
        # Has a default -> written as a live assignment.
        assert "NOOB_SERVER=localhost" in rendered

    def test_override_is_written(self) -> None:
        rendered = render_base({"START_STAGE": "1b"})
        assert "START_STAGE=1b" in rendered
        assert "# START_STAGE=" not in rendered

    def test_sections_present(self) -> None:
        rendered = render_base({})
        for group in EnvGroup:
            assert f"# --- {group.value} ---" in rendered


class TestCheckTargeting:
    """Deciding whether a path variable is checked locally or on NOOB_SERVER."""

    @pytest.mark.parametrize("raw", [None, "", "localhost", "local", "LOCALHOST"])
    def test_local_server_aliases_resolve_to_none(self, raw: str | None) -> None:
        assert _resolve_server(raw) is None

    def test_remote_server_is_stripped(self) -> None:
        assert _resolve_server("  icrhome08  ") == "icrhome08"

    def test_schema_fields_use_declared_group(self) -> None:
        assert _group_of("DATA_ROOT_PATH") is EnvGroup.storage
        assert _group_of("CRDS_PATH") is EnvGroup.setup
        assert _group_of("NOOBOX_PATH") is EnvGroup.noob

    def test_unknown_path_var_defaults_to_storage(self) -> None:
        # The dynamic STAGE_<STAGE>_PATH family is not in the schema.
        assert _group_of("STAGE_2A_PATH") is EnvGroup.storage

    def test_only_storage_paths_go_remote(self) -> None:
        assert _is_remote_path("DATA_ROOT_PATH", "icrhome08")
        assert _is_remote_path("STAGE_2A_PATH", "icrhome08")
        # Non-storage groups stay local even with a remote server.
        assert not _is_remote_path("CRDS_PATH", "icrhome08")
        assert not _is_remote_path("NOOBOX_PATH", "icrhome08")

    def test_nothing_is_remote_without_a_server(self) -> None:
        assert not _is_remote_path("DATA_ROOT_PATH", None)
        assert not _is_remote_path("STAGE_2A_PATH", None)

    def test_mkdir_creates_noobox_parent_not_leaf(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NOOBOX_PATH", raising=False)
        manifest = tmp_path / "box" / "noobox.json"
        env_file = tmp_path / ".env"
        env_file.write_text(f"NOOBOX_PATH={manifest}\n")

        cli_check(mkdir=True, env_file=env_file)

        assert manifest.parent.is_dir()
        assert not manifest.exists()


class TestStageDir:
    """Auto-naming a stage directory under the data root."""

    def test_groups_by_leading_digit(self) -> None:
        assert stage_dir("/data", "1b") == "/data/stage1/1b"
        assert stage_dir("/data", "2a") == "/data/stage2/2a"

    def test_custom_stage_labels(self) -> None:
        # Custom reduction steps like 2bi/2bii group under stage2.
        assert stage_dir("/data", "2bi") == "/data/stage2/2bi"
        assert stage_dir("/data", "2bii") == "/data/stage2/2bii"


class TestFindOverrides:
    """Detecting shell variables that shadow the .env declarations."""

    def test_no_shell_no_override(self) -> None:
        assert find_overrides({"NOOB_SERVER": "a"}, {}) == []

    def test_differing_shell_value_is_an_override(self) -> None:
        assert find_overrides({"NOOB_SERVER": "a"}, {"NOOB_SERVER": "b"}) == [
            ("NOOB_SERVER", "a", "b")
        ]

    def test_matching_shell_value_is_not_an_override(self) -> None:
        assert find_overrides({"NOOB_SERVER": "a"}, {"NOOB_SERVER": "a"}) == []

    def test_unset_declaration_is_skipped(self) -> None:
        assert find_overrides({"NOOB_SERVER": None}, {"NOOB_SERVER": "b"}) == []


class TestSplitExisting:
    """Splitting a path into its existing prefix and missing remainder."""

    def test_fully_existing(self, tmp_path: Path) -> None:
        existing, missing = split_existing(tmp_path)
        assert existing == tmp_path
        assert missing is None

    def test_partially_existing(self, tmp_path: Path) -> None:
        existing, missing = split_existing(tmp_path / "nope" / "deep")
        assert existing == tmp_path
        assert missing == Path("nope/deep")


class TestStartStageCallback:
    """Validation/normalisation of a single stage label."""

    def test_lowercases(self) -> None:
        assert start_stage_callback("2BI") == "2bi"

    def test_none_and_empty_pass_through(self) -> None:
        assert start_stage_callback(None) is None
        assert start_stage_callback("") is None

    def test_rejects_non_digit_start(self) -> None:
        with pytest.raises(typer.BadParameter):
            start_stage_callback("b1")


class TestStageListCallback:
    """Validation/normalisation of a list of stage labels."""

    def test_lowercases_all(self) -> None:
        assert stage_list_callback(["1B", "2A", "2Bi"]) == ["1b", "2a", "2bi"]

    def test_rejects_non_digit_start(self) -> None:
        with pytest.raises(typer.BadParameter):
            stage_list_callback(["1b", "xx"])
