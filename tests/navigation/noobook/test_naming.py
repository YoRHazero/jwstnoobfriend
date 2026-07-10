"""Tests for JWST product-name parsing in ``noobfriend.navigation.noobook._naming``."""

from noobfriend.navigation.noobook._naming import (
    parse_exposure_name,
    parse_program_id,
    strip_fits_suffix,
)


def test_strip_fits_suffix_removes_known_extensions():
    assert (
        strip_fits_suffix("jw01345001001_02201_00001_nrca1_cal.fits")
        == "jw01345001001_02201_00001_nrca1_cal"
    )
    assert strip_fits_suffix("x_cal.fits.gz") == "x_cal"
    assert strip_fits_suffix("already_stem") == "already_stem"


def test_parse_exposure_name_fields():
    parsed = parse_exposure_name("jw01345001001_02201_00001_nrca1_cal.fits")
    assert parsed is not None
    assert parsed.program_id == "01345"
    assert parsed.observation == "001"
    assert parsed.visit == "001"
    assert parsed.ggsaa == "02201"
    assert parsed.exposure == "00001"
    assert parsed.detector == "nrca1"
    assert parsed.suffix == "cal"


def test_parse_exposure_name_keeps_multipart_suffix():
    parsed = parse_exposure_name("jw01345001001_02201_00001_nrcalong_rateints.fits")
    assert parsed is not None
    assert parsed.detector == "nrcalong"
    assert parsed.suffix == "rateints"


def test_parse_exposure_name_rejects_stage3_name():
    assert parse_exposure_name("jw01345-o001_t021_nircam_f200w_i2d.fits") is None


def test_parse_program_id_handles_exposure_and_stage3():
    assert parse_program_id("jw01345001001_02201_00001_nrca1_rate.fits") == "01345"
    assert parse_program_id("jw01345-o001_t021_nircam_f200w_i2d.fits") == "01345"
    assert parse_program_id("not_a_jwst_file.fits") is None
