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
    product_level_callback,
    proposal_id_callback,
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
