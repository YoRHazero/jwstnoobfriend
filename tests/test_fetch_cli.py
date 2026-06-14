"""Unit tests for the pure logic of the fetch CLI (options + manifest service)."""

import pytest
import typer

from noobfriend.cli.fetch._manifest_service import (
    count_manifest_columns,
    deduplicate_columns,
    filter_manifest_terms,
    normalize_manifest_value,
    parse_filter_value,
    summarize_manifest_column,
    validate_columns,
)
from noobfriend.cli.fetch._options import (
    DEFAULT_PRODUCT_LEVEL,
    product_level_callback,
    proposal_id_callback,
    resolve_product_level,
)
from noobfriend.cli.fetch._search_service import (
    is_fits_product,
    is_rateint_product,
)


class TestProposalIdCallback:
    """Normalization and validation of proposal IDs."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1895", "01895"),
            ("01895", "01895"),
            ("  1895  ", "01895"),
            ("0001895", "01895"),
            ("12345", "12345"),
        ],
    )
    def test_normalizes_to_five_digits(self, raw: str, expected: str) -> None:
        assert proposal_id_callback(raw) == expected

    @pytest.mark.parametrize("raw", ["abc", "12a", "", "0"])
    def test_rejects_non_numeric(self, raw: str) -> None:
        with pytest.raises(typer.BadParameter):
            proposal_id_callback(raw)

    def test_rejects_too_long(self) -> None:
        with pytest.raises(typer.BadParameter):
            proposal_id_callback("123456")


class TestProductLevelCallback:
    """Validation of the product-level option."""

    @pytest.mark.parametrize("level", ["1b", "2a", "2b", "2c"])
    def test_accepts_known_levels(self, level: str) -> None:
        assert product_level_callback(level) == level

    @pytest.mark.parametrize("level", ["3a", "1B", "", "two"])
    def test_rejects_unknown_levels(self, level: str) -> None:
        with pytest.raises(typer.BadParameter):
            product_level_callback(level)

    def test_none_passes_through(self) -> None:
        # An omitted option resolves later from START_STAGE.
        assert product_level_callback(None) is None


class TestResolveProductLevel:
    """Precedence of --product-level > START_STAGE > default."""

    def test_cli_value_wins(self) -> None:
        # An explicit value is used even when START_STAGE is set (and ignored
        # even if START_STAGE would be invalid).
        assert resolve_product_level("2b", "2a") == "2b"
        assert resolve_product_level("2b", "3a") == "2b"

    def test_falls_back_to_env(self) -> None:
        assert resolve_product_level(None, "2a") == "2a"

    def test_env_is_normalized(self) -> None:
        assert resolve_product_level(None, "  2A  ") == "2a"

    def test_defaults_when_nothing_set(self) -> None:
        assert resolve_product_level(None, None) == DEFAULT_PRODUCT_LEVEL
        assert resolve_product_level(None, "") == DEFAULT_PRODUCT_LEVEL

    @pytest.mark.parametrize("env_value", ["3a", "2bi", "two"])
    def test_invalid_env_raises(self, env_value: str) -> None:
        with pytest.raises(typer.BadParameter):
            resolve_product_level(None, env_value)


class TestParseFilterValue:
    """JSON-literal parsing of raw filter values."""

    def test_parses_json_literals(self) -> None:
        assert parse_filter_value("2") == 2
        assert parse_filter_value("true") is True
        assert parse_filter_value("null") is None

    def test_falls_back_to_raw_string(self) -> None:
        assert parse_filter_value("NIRCAM") == "NIRCAM"
        assert parse_filter_value("jw01895") == "jw01895"


class TestNormalizeManifestValue:
    """Canonical normalization of manifest values."""

    def test_integer_and_parsed_filter_match(self) -> None:
        # A column holding the int 2 must match a "--value 2" filter.
        assert normalize_manifest_value(2) == normalize_manifest_value(
            parse_filter_value("2")
        )

    def test_string_is_quoted(self) -> None:
        assert normalize_manifest_value("NIRCAM") == '"NIRCAM"'

    def test_dict_key_order_is_canonical(self) -> None:
        assert normalize_manifest_value({"a": 1, "b": 2}) == normalize_manifest_value(
            {"b": 2, "a": 1}
        )


class TestManifestColumns:
    """Column counting, summarizing, deduplication, and validation."""

    def test_count_columns_counts_presence_per_term(self) -> None:
        products = [
            {"a": 1, "b": 2},
            {"a": 3},
            {"c": 4},
        ]
        counts = count_manifest_columns(products)
        assert counts == {"a": 2, "b": 1, "c": 1}

    def test_summarize_column_counts_values(self) -> None:
        products = [
            {"instrument": "NIRCAM"},
            {"instrument": "NIRCAM"},
            {"instrument": "MIRI"},
            {"other": "x"},
        ]
        counts = summarize_manifest_column(products, "instrument")
        assert counts == {'"NIRCAM"': 2, '"MIRI"': 1}

    def test_deduplicate_preserves_first_seen_order(self) -> None:
        assert deduplicate_columns(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]

    def test_validate_columns_passes_for_known(self) -> None:
        available = count_manifest_columns([{"a": 1, "b": 2}])
        validate_columns(["a", "b"], available)  # should not raise

    def test_validate_columns_raises_for_unknown(self) -> None:
        available = count_manifest_columns([{"a": 1}])
        with pytest.raises(typer.BadParameter):
            validate_columns(["a", "missing"], available)


class TestFilterManifestTerms:
    """Filtering manifest terms by column values."""

    def test_selects_matching_string_values(self) -> None:
        products = [
            {"instrument": "NIRCAM", "id": 1},
            {"instrument": "MIRI", "id": 2},
            {"instrument": "NIRCAM", "id": 3},
        ]
        selected = filter_manifest_terms(products, "instrument", ["NIRCAM"])
        assert [term["id"] for term in selected] == [1, 3]

    def test_matches_numeric_values_via_json_parsing(self) -> None:
        products = [{"level": 1}, {"level": 2}, {"level": 2}]
        selected = filter_manifest_terms(products, "level", ["2"])
        assert selected == [{"level": 2}, {"level": 2}]

    def test_supports_multiple_values(self) -> None:
        products = [
            {"instrument": "NIRCAM"},
            {"instrument": "MIRI"},
            {"instrument": "NIRSPEC"},
        ]
        selected = filter_manifest_terms(products, "instrument", ["NIRCAM", "MIRI"])
        assert selected == [{"instrument": "NIRCAM"}, {"instrument": "MIRI"}]

    def test_skips_terms_missing_the_column(self) -> None:
        products = [{"instrument": "NIRCAM"}, {"other": "x"}]
        selected = filter_manifest_terms(products, "instrument", ["NIRCAM"])
        assert selected == [{"instrument": "NIRCAM"}]


class TestIsRateintProduct:
    """Recognition of stage-2a per-integration (rateints) products."""

    def test_matches_rateints_fits_by_file_suffix(self) -> None:
        product = {
            "filename": "jw01895001001_02101_00001_nrcb1_rateints.fits",
            "file_suffix": "_rateints",
        }
        assert is_rateint_product(product) is True

    def test_matches_rateints_preview_via_filename(self) -> None:
        # The preview's file_suffix is the generic "_preview"; only the
        # filename token identifies it as a rateints preview.
        product = {
            "filename": "jw01895001001_02101_00001_nrcb1_rateints.jpg",
            "file_suffix": "_preview",
        }
        assert is_rateint_product(product) is True

    def test_does_not_match_rate_image(self) -> None:
        # "rate" is a prefix of "rateints" but must not be matched.
        product = {
            "filename": "jw01895001001_02101_00001_nrcb1_rate.fits",
            "file_suffix": "_rate",
        }
        assert is_rateint_product(product) is False

    def test_does_not_match_other_suffixes(self) -> None:
        for suffix in ("_uncal", "_cal", "_thumb"):
            product = {
                "filename": f"jw01895001001_02101_00001_nrcb1{suffix}.fits",
                "file_suffix": suffix,
            }
            assert is_rateint_product(product) is False

    def test_tolerates_missing_fields(self) -> None:
        assert is_rateint_product({}) is False
        assert is_rateint_product({"file_suffix": "RATEINTS"}) is True


class TestIsFitsProduct:
    """Recognition of FITS data files versus preview/thumbnail images."""

    @pytest.mark.parametrize(
        "filename",
        [
            "jw01895001001_02101_00001_nrcb1_rate.fits",
            "jw01895001001_02101_00001_nrcb1_rate.FITS",
            "x_cal.fits.gz",
        ],
    )
    def test_matches_fits_files(self, filename: str) -> None:
        assert is_fits_product({"filename": filename}) is True

    @pytest.mark.parametrize(
        "filename",
        [
            "jw01895001001_02101_00001_nrcb1_rate.jpg",
            "jw01895001001_02101_00001_nrcb1_thumb.jpg",
        ],
    )
    def test_rejects_preview_and_thumbnail(self, filename: str) -> None:
        assert is_fits_product({"filename": filename}) is False

    def test_tolerates_missing_filename(self) -> None:
        assert is_fits_product({}) is False
