"""Unit tests for the pure logic of remote downloads (parsing + script gen)."""

import pytest

from noobfriend.cli.fetch._remote_service import (
    build_remote_script,
    parse_destination,
    parse_progress_line,
)


class TestParseDestination:
    """scp-style disambiguation of [user@]host:path versus local paths."""

    def test_remote_with_absolute_path(self) -> None:
        target = parse_destination("icrhome08:/data/jwst/raw")
        assert target.is_remote
        assert target.ssh_dest == "icrhome08"
        assert target.path == "/data/jwst/raw"

    def test_remote_with_user_and_relative_path(self) -> None:
        target = parse_destination("zchao@icrhome08:rel/dir")
        assert target.is_remote
        assert target.ssh_dest == "zchao@icrhome08"
        assert target.path == "rel/dir"

    def test_remote_empty_path_defaults_to_dot(self) -> None:
        target = parse_destination("icrhome08:")
        assert target.is_remote
        assert target.path == "."

    @pytest.mark.parametrize("spec", ["/data/raw", "./downloads", "downloads"])
    def test_local_paths(self, spec: str) -> None:
        target = parse_destination(spec)
        assert not target.is_remote
        assert target.ssh_dest is None
        assert target.path == spec

    def test_colon_after_slash_is_local(self) -> None:
        # A ':' that appears after a '/' belongs to the path, not a host.
        target = parse_destination("/weird/a:b")
        assert not target.is_remote
        assert target.path == "/weird/a:b"

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_destination(":/data/raw")


class TestParseProgressLine:
    """Parsing the remote worker's DONE/SKIP/FAIL status lines."""

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("DONE jw01895_a.fits", ("DONE", "jw01895_a.fits")),
            ("SKIP jw01895_b.fits\n", ("SKIP", "jw01895_b.fits")),
            ("  FAIL jw01895_c.fits  ", ("FAIL", "jw01895_c.fits")),
        ],
    )
    def test_recognised_lines(self, line: str, expected: tuple[str, str]) -> None:
        assert parse_progress_line(line) == expected

    @pytest.mark.parametrize("line", ["", "junk", "DONEx.fits", "downloading foo"])
    def test_unrecognised_lines(self, line: str) -> None:
        assert parse_progress_line(line) is None


class TestBuildRemoteScript:
    """Server-side bash script generation for remote-direct downloads."""

    products = [
        {"filename": "jw01895_a_uncal.fits", "size": 100},
        {"filename": "jw01895_b_uncal.fits", "size": 200},
    ]

    def test_embeds_dir_url_and_listing(self) -> None:
        script = build_remote_script(
            self.products, "/data/jwst/raw", skip_exist=False, concurrency=4
        )
        assert "DIR=/data/jwst/raw" in script
        assert "product_name=$name" in script
        assert "xargs -P 4 -n2" in script
        assert "jw01895_a_uncal.fits 100" in script
        assert "jw01895_b_uncal.fits 200" in script

    def test_skip_flag_toggles(self) -> None:
        with_skip = build_remote_script(
            self.products, "/d", skip_exist=True, concurrency=2
        )
        without_skip = build_remote_script(
            self.products, "/d", skip_exist=False, concurrency=2
        )
        assert "SKIP=1" in with_skip
        assert "SKIP=''" in without_skip

    def test_quotes_remote_dir_with_spaces(self) -> None:
        script = build_remote_script(
            self.products, "/data/my raw", skip_exist=False, concurrency=2
        )
        assert "DIR='/data/my raw'" in script
