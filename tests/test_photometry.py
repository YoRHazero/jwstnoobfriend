"""Tests for native-grid multi-band aperture photometry and SED plotting."""

from types import SimpleNamespace
from typing import Any, Callable

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from noobfriend.extraction.photometry import ApertureSED
from noobfriend.extraction.photometry._aperture import grow_aperture_mask
from noobfriend.extraction.photometry._band import normalize_band
from noobfriend.extraction.photometry._core import _auto_label_map
from noobfriend.extraction.photometry._coverage import reproject_coverage
from noobfriend.extraction.photometry._floor import _generic_floor, _label_floor
from noobfriend.extraction.photometry._measure import (
    BandPhotometry,
    aperture_snr,
    measure_band,
)
from noobfriend.extraction.photometry._noise import measure_aperture_noise
from noobfriend.extraction.photometry._plot import sed_figure, sed_figure_mpl
from noobfriend.extraction.photometry._thumbnails import _aperture_bbox


class LinearWCS:
    """A small affine WCS stand-in supporting detector/world transforms.

    ``world = (pixel - origin) * scale + ref``, so ``scale`` is degrees per
    pixel and a larger ``scale`` is a coarser grid.
    """

    available_frames = ["detector", "world"]

    def __init__(
        self,
        *,
        x0: float = 0.0,
        y0: float = 0.0,
        ra0: float = 0.0,
        dec0: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.ra0 = ra0
        self.dec0 = dec0
        self.scale = scale

    def get_transform(
        self, from_frame: str, to_frame: str
    ) -> Callable[[Any, Any], tuple[Any, Any]]:
        """Return detector/world transforms for the requested direction."""
        if (from_frame, to_frame) == ("world", "detector"):
            return self._world_to_detector
        if (from_frame, to_frame) == ("detector", "world"):
            return self._detector_to_world
        raise ValueError(f"Unsupported transform: {from_frame!r} -> {to_frame!r}.")

    def _world_to_detector(self, ra: Any, dec: Any) -> tuple[Any, Any]:
        return (
            (ra - self.ra0) / self.scale + self.x0,
            (dec - self.dec0) / self.scale + self.y0,
        )

    def _detector_to_world(self, x: Any, y: Any) -> tuple[Any, Any]:
        return (
            (x - self.x0) * self.scale + self.ra0,
            (y - self.y0) * self.scale + self.dec0,
        )


def _sed(bands: dict[str, Any], **kwargs: Any) -> ApertureSED:
    """Build an ApertureSED with a fixed test identity."""
    return ApertureSED("TEST", 0.0, 0.0, bands=bands, **kwargs)


def _label(shape: tuple[int, int], pixels: set[tuple[int, int]]) -> np.ndarray:
    """Build a segmentation map with label 1 at ``(row, col)`` pixels."""
    labels = np.zeros(shape, dtype=np.int32)
    for row, col in pixels:
        labels[row, col] = 1
    return labels


def _band(
    data: np.ndarray,
    *,
    wcs: LinearWCS | None = None,
    wavelength: float = 1.0,
    wavelength_error: tuple[float, ...] | None = None,
    flux_scale_mjy: float | None = None,
    flux_unit: str | None = None,
    label_pixels: set[tuple[int, int]] | None = None,
    allow_background: bool = False,
    error: np.ndarray | None = None,
) -> dict[str, object]:
    """Build one public band spec for tests.

    ``allow_background`` defaults to ``False`` here (unlike the library default
    of ``True``) so labelled tests stay hard-confined to their segment and their
    deterministic flux assertions hold.
    """
    spec: dict[str, object] = {
        "data": np.asarray(data, dtype=float),
        "wcs": wcs or LinearWCS(),
        "wavelength": wavelength,
        "allow_background": allow_background,
    }
    if wavelength_error is not None:
        spec["wavelength_error"] = wavelength_error
    if flux_scale_mjy is not None:
        spec["flux_scale_mjy"] = flux_scale_mjy
    if flux_unit is not None:
        spec["flux_unit"] = flux_unit
    if label_pixels is not None:
        spec["label_map"] = _label(np.asarray(data).shape, label_pixels)
    if error is not None:
        spec["error"] = np.asarray(error, dtype=float)
    return spec


def _grow_all_label_kwargs() -> dict[str, object]:
    """Disable stop criteria so noobase fills the allowed label region."""
    return {
        "snr_threshold": None,
        "gradient_ratio_threshold": 1e99,
    }


def _block(rows: range, cols: range) -> set[tuple[int, int]]:
    """Return the set of ``(row, col)`` pixels in a rectangular block."""
    return {(r, c) for r in rows for c in cols}


class TestApertureSEDInputs:
    """Public input normalization and validation."""

    def test_wavelength_error_must_be_tuple(self) -> None:
        with pytest.raises(ValueError, match="wavelength_error must be a tuple"):
            _sed(
                {
                    "F090W": {
                        "data": np.ones((3, 3)),
                        "wcs": LinearWCS(),
                        "wavelength": 0.9,
                        "wavelength_error": 0.05,
                    }
                }
            )

    def test_wavelength_error_len_one_is_symmetric(self) -> None:
        data = np.ones((3, 3))
        sed = _sed(
            {
                "F090W": _band(
                    data,
                    wavelength=0.9,
                    wavelength_error=(0.05,),
                    label_pixels={(1, 1)},
                )
            }
        )

        result = sed.draft(
            seed_xy_by_band={"F090W": (1.0, 1.0)},
            grow_kwargs=_grow_all_label_kwargs(),
        ).measure()

        assert result.measurements[0].wavelength_error == (0.05, 0.05)

    def test_wavelength_error_other_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="length 1 or 2"):
            _sed(
                {
                    "F090W": _band(
                        np.ones((3, 3)),
                        wavelength=0.9,
                        wavelength_error=(0.01, 0.02, 0.03),
                    )
                }
            )

    def test_empty_bands_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one band"):
            ApertureSED("TEST", 0.0, 0.0)


class TestOptionalWCS:
    """WCS-less bands get a synthesised tangent-plane WCS from the cutout size."""

    def test_unknown_band_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown key"):
            ApertureSED(
                "T",
                10.0,
                20.0,
                bands={
                    "A": {
                        "data": np.ones((3, 3)),
                        "wcs": LinearWCS(),
                        "lable_map": None,  # typo for label_map
                    }
                },
            )

    def test_no_wcs_no_size_raises(self) -> None:
        with pytest.raises(ValueError, match="no 'wcs'"):
            ApertureSED("T", 10.0, 20.0, bands={"A": {"data": np.ones((5, 5))}})

    def test_synthesized_wcs_seeds_at_centre(self) -> None:
        sed = ApertureSED(
            "T",
            10.0,
            20.0,
            bands={"A": {"data": np.ones((5, 5)), "wavelength": 1.0}},
            cutout_size_arcsec=5.0,
        )
        draft = sed.draft(grow_kwargs=_grow_all_label_kwargs())
        # (ra, dec) maps to the centre pixel (2, 2) of the 5x5 image.
        assert draft.source_apertures["A"].seed_xy == (2, 2)
        result = draft.measure(correlated_error=False)
        assert bool(result.union_mask.any())
        assert bool(result.union_mask[2, 2])  # the union grows from the centre seed

    def test_synthesized_wcs_pixel_scale(self) -> None:
        from noobfriend.extraction._wcs import (
            pixel_scale_per_deg,
            tangent_plane_wcs,
            world_detector_transforms,
        )

        wcs = tangent_plane_wcs(10.0, 20.0, size_arcsec=5.0, shape=(10, 10))
        _, detector_to_world = world_detector_transforms(wcs)
        scale_x, scale_y = pixel_scale_per_deg(detector_to_world, 5, 5)
        # 10 px over 5" -> 0.5"/px -> 3600 / 0.5 = 7200 px/deg.
        assert scale_x == pytest.approx(7200.0, rel=1e-3)
        assert scale_y == pytest.approx(7200.0, rel=1e-3)

    def test_two_wcsless_bands_share_a_union(self) -> None:
        # Co-centred cutouts at two resolutions, both WCS synthesised.
        sed = ApertureSED(
            "T",
            10.0,
            20.0,
            bands={
                "fine": {
                    "data": np.ones((8, 8)),
                    "wavelength": 1.0,
                    "flux_scale_mjy": 0.01,
                },
                "coarse": {
                    "data": np.full((4, 4), 4.0),
                    "wavelength": 2.0,
                    "flux_scale_mjy": 0.01,
                },
            },
            cutout_size_arcsec=4.0,
            reference="finest",
        )
        result = sed.draft(grow_kwargs=_grow_all_label_kwargs()).measure(
            correlated_error=False
        )
        by_band = {m.band: m for m in result.measurements}
        assert result.reference_band == "fine"  # the smaller pixels
        assert by_band["fine"].flux_mjy > 0
        assert by_band["coarse"].flux_mjy > 0


class TestApertureSEDMeasure:
    """Union-aperture measurement behaviour."""

    def test_single_band_native_sum(self) -> None:
        sed = _sed(
            {
                "F090W": _band(
                    np.ones((5, 5)),
                    wavelength=0.9,
                    flux_scale_mjy=0.01,
                    flux_unit="test-unit",
                    label_pixels={(2, 1), (2, 2), (2, 3)},
                    error=np.ones((5, 5)),
                )
            }
        )

        result = sed.draft(
            seed_xy_by_band={"F090W": (2.0, 2.0)},
            grow_kwargs=_grow_all_label_kwargs(),
        ).measure()

        m = result.measurements[0]
        assert m.flux == pytest.approx(3.0)
        assert m.flux_mjy == pytest.approx(0.03)
        assert m.error == pytest.approx(np.sqrt(3.0))
        assert m.error_mjy == pytest.approx(0.01 * np.sqrt(3.0))
        assert m.flux_unit == "test-unit"
        assert m.covered_area == pytest.approx(3.0)
        assert m.valid_area == pytest.approx(3.0)
        assert m.bad_fraction == pytest.approx(0.0)
        assert not m.flagged

    def test_result_carries_source_identity(self) -> None:
        sed = ApertureSED(
            "GOODS-1",
            12.5,
            -7.25,
            bands={"F090W": _band(np.ones((3, 3)), label_pixels={(1, 1)})},
        )
        result = sed.draft(
            seed_xy_by_band={"F090W": (1.0, 1.0)},
            grow_kwargs=_grow_all_label_kwargs(),
        ).measure()
        assert result.source_id == "GOODS-1"
        assert (result.ra, result.dec) == (12.5, -7.25)

    def test_union_flags_band_with_nan_in_aperture(self) -> None:
        a = np.ones((5, 5))
        b = np.ones((5, 5))
        b[2, 1] = np.nan  # In the union via band A, missing for band B.
        err = np.ones((5, 5))

        sed = _sed(
            {
                "F090W": _band(
                    a,
                    wavelength=0.9,
                    flux_scale_mjy=0.01,
                    label_pixels={(2, 1), (2, 2)},
                    error=err,
                ),
                "F115W": _band(
                    b,
                    wavelength=1.15,
                    label_pixels={(2, 2), (2, 3)},
                    error=err,
                ),
            },
            reference="F090W",
            flag_bad_fraction=0.10,
        )

        result = sed.draft(
            seed_xy_by_band={"F090W": (2.0, 2.0), "F115W": (2.0, 2.0)},
            grow_kwargs=_grow_all_label_kwargs(),
        ).measure()

        assert result.reference_band == "F090W"
        assert int(result.union_mask.sum()) == 3
        by_band = {m.band: m for m in result.measurements}
        assert by_band["F090W"].flux == pytest.approx(3.0)
        assert by_band["F090W"].error == pytest.approx(np.sqrt(3.0))
        assert by_band["F090W"].bad_fraction == pytest.approx(0.0)
        assert not by_band["F090W"].flagged
        assert by_band["F115W"].flux == pytest.approx(2.0)
        assert by_band["F115W"].valid_area == pytest.approx(2.0)
        assert by_band["F115W"].covered_area == pytest.approx(3.0)
        assert by_band["F115W"].bad_fraction == pytest.approx(1 / 3)
        assert by_band["F115W"].flagged

    def test_reference_finest_selects_smallest_pixel_scale_band(self) -> None:
        sed = _sed(
            {
                "coarse": _band(
                    np.ones((5, 5)),
                    wcs=LinearWCS(scale=2.0),
                    wavelength=2.0,
                    label_pixels={(2, 2)},
                ),
                "fine": _band(
                    np.ones((5, 5)),
                    wcs=LinearWCS(scale=1.0),
                    wavelength=1.0,
                    label_pixels={(2, 2)},
                ),
            },
            reference="finest",
        )

        result = sed.draft(
            seed_world=(2.0, 2.0),
            grow_kwargs=_grow_all_label_kwargs(),
        ).measure()

        assert result.reference_band == "fine"
        assert result.union_mask.shape == (5, 5)

    def test_seed_defaults_to_source_position(self) -> None:
        # Source at (ra, dec) = (2, 2) maps to pixel (2, 2) under the identity WCS.
        sed = ApertureSED(
            "TEST",
            2.0,
            2.0,
            bands={"F090W": _band(np.ones((5, 5)), label_pixels={(2, 2)})},
        )
        result = sed.draft(grow_kwargs=_grow_all_label_kwargs()).measure()
        assert int(result.union_mask.sum()) == 1

    def test_seed_keys_must_match_bands(self) -> None:
        sed = _sed({"F090W": _band(np.ones((3, 3)), label_pixels={(1, 1)})})

        with pytest.raises(ValueError, match="keys must match band names"):
            sed.draft(
                seed_xy_by_band={"F115W": (1.0, 1.0)},
                grow_kwargs=_grow_all_label_kwargs(),
            )


class TestApertureSEDBuild:
    """The synchronous ``build`` assembler and its database merge."""

    def test_merge_prefers_existing_band(self, monkeypatch: Any) -> None:
        async def fake_loader(ra: float, dec: float, **kwargs: Any) -> Any:
            return SimpleNamespace(
                bands={
                    "f090w": _band(np.ones((5, 5)), wavelength=0.9),
                    "f150w": _band(np.ones((5, 5)), wavelength=1.5),
                },
                metadata={},
            )

        monkeypatch.setattr("noobfriend.core.io.load_grizli_cutout", fake_loader)

        sed = ApertureSED.build(
            "S",
            0.0,
            0.0,
            bands={"f090w": _band(np.ones((3, 3)), wavelength=0.9)},
            extra_database="grizli",
        )

        assert set(sed._by_name) == {"f090w", "f150w"}
        # The directly-provided band wins on conflict (3x3, not the fetched 5x5).
        assert sed._by_name["f090w"].data.shape == (3, 3)
        assert sed._by_name["f150w"].data.shape == (5, 5)

    def test_rejects_unknown_database(self) -> None:
        with pytest.raises(ValueError, match="Unsupported extra_database"):
            ApertureSED.build(
                "S",
                0.0,
                0.0,
                bands={"a": _band(np.ones((3, 3)))},
                extra_database="sdss",
            )


class TestApertureSEDDraft:
    """The inspectable draft: union selection and per-band re-growth."""

    def _two_band_draft(self) -> Any:
        sed = _sed(
            {
                "A": _band(
                    np.ones((5, 5)),
                    wavelength=1.0,
                    label_pixels={(2, 1), (2, 2)},
                ),
                "B": _band(
                    np.ones((5, 5)),
                    wavelength=2.0,
                    label_pixels={(2, 2), (2, 3)},
                ),
            },
            reference="A",
        )
        return sed.draft(
            seed_xy_by_band={"A": (2.0, 2.0), "B": (2.0, 2.0)},
            grow_kwargs=_grow_all_label_kwargs(),
        )

    def test_union_bands_selects_subset(self) -> None:
        draft = self._two_band_draft()
        assert int(draft.union_mask().sum()) == 3  # A and B apertures combined
        assert int(draft.union_mask(union_bands=["A"]).sum()) == 2  # A only

    def test_union_bands_unknown_raises(self) -> None:
        draft = self._two_band_draft()
        with pytest.raises(ValueError, match="unknown union band"):
            draft.union_mask(union_bands=["Z"])

    def test_measure_through_subset_union(self) -> None:
        draft = self._two_band_draft()
        result = draft.measure(union_bands=["A"], correlated_error=False)
        # Both bands are still measured, but through A's 2-pixel union.
        by_band = {m.band: m for m in result.measurements}
        assert by_band["A"].covered_area == pytest.approx(2.0)
        assert by_band["B"].covered_area == pytest.approx(2.0)

    def test_regrow_replaces_one_band(self) -> None:
        sed = _sed({"A": _band(np.ones((5, 5)), wavelength=1.0)})
        draft = sed.draft(
            seed_xy_by_band={"A": (2.0, 2.0)},
            grow_kwargs=_grow_all_label_kwargs(),
        )
        before = int(draft.source_apertures["A"].mask.sum())

        labels = _label((5, 5), {(2, 2)})
        regrown = draft.regrow("A", label_map=labels, allow_background=False)

        assert int(regrown.source_apertures["A"].mask.sum()) == 1  # confined
        assert before > 1
        # The original draft is untouched (frozen, returns a new instance).
        assert int(draft.source_apertures["A"].mask.sum()) == before

    def test_regrow_unknown_band_raises(self) -> None:
        draft = self._two_band_draft()
        with pytest.raises(ValueError, match="unknown band"):
            draft.regrow("Z")

    def test_plot_apertures_previews_before_measure(self) -> None:
        draft = self._two_band_draft()
        fig = draft.plot_apertures()
        titled = [ax for ax in fig.axes if ax.get_title()]
        assert len(titled) == 2


class TestPlotSegmentation:
    """Side-by-side image / segmentation diagnostic (draft live, result used)."""

    def _blob_sed(self) -> ApertureSED:
        data = np.zeros((9, 9))
        data[3:6, 3:6] = 1.0  # a detectable source at the seed
        return _sed({"A": _band(data, wavelength=1.0)})

    def test_draft_segmentation_renders_two_panels(self) -> None:
        draft = self._blob_sed().draft(
            seed_xy_by_band={"A": (4.0, 4.0)}, grow_kwargs=_grow_all_label_kwargs()
        )
        fig = draft.plot_segmentation("A")
        assert len(fig.axes) == 2  # image | labels

    def test_draft_segmentation_unknown_band(self) -> None:
        draft = self._blob_sed().draft(
            seed_xy_by_band={"A": (4.0, 4.0)}, grow_kwargs=_grow_all_label_kwargs()
        )
        with pytest.raises(ValueError, match="unknown band"):
            draft.plot_segmentation("Z")

    def test_result_shows_used_label_map(self) -> None:
        result = (
            self._blob_sed()
            .draft(
                seed_xy_by_band={"A": (4.0, 4.0)},
                auto_segment=True,
                grow_kwargs=_grow_all_label_kwargs(),
            )
            .measure(correlated_error=False)
        )
        assert result.label_maps["A"] is not None
        fig = result.plot_segmentation("A")
        assert len(fig.axes) == 2

    def test_result_raises_when_unconstrained(self) -> None:
        result = (
            self._blob_sed()
            .draft(
                seed_xy_by_band={"A": (4.0, 4.0)},
                grow_kwargs=_grow_all_label_kwargs(),
            )
            .measure(correlated_error=False)
        )
        assert result.label_maps["A"] is None
        with pytest.raises(ValueError, match="without a segmentation map"):
            result.plot_segmentation("A")


class TestAutoSegment:
    """Auto-segmentation label resolution and its precedence."""

    def test_auto_label_map_detected(self) -> None:
        data = np.zeros((9, 9))
        data[3:6, 3:6] = 1.0
        lm = _auto_label_map(data, (4.0, 4.0), None)
        assert lm is not None
        assert lm[4, 4] > 0

    def test_auto_label_map_not_detected(self) -> None:
        data = np.zeros((9, 9))
        data[0:3, 0:3] = 1.0  # source in the corner, seed on background
        assert _auto_label_map(data, (5.0, 5.0), None) is None

    def test_auto_segment_attaches_only_when_unconstrained(self) -> None:
        data = np.zeros((9, 9))
        data[3:6, 3:6] = 1.0
        sed = _sed({"A": _band(data, wavelength=1.0)})
        on = sed.draft(
            seed_xy_by_band={"A": (4.0, 4.0)},
            auto_segment=True,
            grow_kwargs=_grow_all_label_kwargs(),
        )
        off = sed.draft(
            seed_xy_by_band={"A": (4.0, 4.0)},
            auto_segment=False,
            grow_kwargs=_grow_all_label_kwargs(),
        )
        assert on.label_maps["A"] is not None
        assert off.label_maps["A"] is None

    def test_override_label_map_wins_over_spec(self) -> None:
        override = _label((5, 5), {(1, 1), (2, 2)})
        sed = _sed({"A": _band(np.ones((5, 5)), wavelength=1.0, label_pixels={(2, 2)})})
        draft = sed.draft(
            seed_xy_by_band={"A": (2.0, 2.0)},
            label_map_by_band={"A": override},
            grow_kwargs=_grow_all_label_kwargs(),
        )
        np.testing.assert_array_equal(draft.label_maps["A"], override)


class TestFluxConservation:
    """Native-grid measurement keeps flux correct across pixel scales."""

    def test_cross_scale_flux_is_conserved(self) -> None:
        # ``coarse`` has 2x the linear pixel scale (4x the solid angle), offset
        # so each coarse pixel is exactly a 2x2 block of fine pixels. A flat
        # source therefore carries 4x the per-pixel value in the coarse band.
        # The measured mJy flux must match: summing surface-brightness values on
        # a common grid (the old behaviour) would make ``coarse`` read 4x high.
        fine = np.ones((8, 8))
        coarse = np.full((4, 4), 4.0)
        sed = _sed(
            {
                "fine": _band(
                    fine,
                    wcs=LinearWCS(scale=1.0),
                    wavelength=1.0,
                    flux_scale_mjy=0.01,
                    label_pixels=_block(range(2, 6), range(2, 6)),
                ),
                "coarse": _band(
                    coarse,
                    wcs=LinearWCS(scale=2.0, ra0=0.5, dec0=0.5),
                    wavelength=2.0,
                    flux_scale_mjy=0.01,
                    label_pixels=_block(range(1, 3), range(1, 3)),
                ),
            },
            reference="finest",
        )

        result = sed.draft(
            seed_world=(3.5, 3.5),
            grow_kwargs=_grow_all_label_kwargs(),
        ).measure()

        by_band = {m.band: m for m in result.measurements}
        assert result.reference_band == "fine"
        # 16 fine pixels * 1.0 * 0.01  ==  4 coarse pixels * 4.0 * 0.01.
        assert by_band["fine"].flux_mjy == pytest.approx(0.16)
        assert by_band["coarse"].flux_mjy == pytest.approx(0.16, rel=1e-6)
        assert by_band["coarse"].covered_area == pytest.approx(4.0, rel=1e-6)

    def test_reproject_coverage_is_fractional(self) -> None:
        # A mask covering only the top fine row of a 2x2 fine block should give
        # the enclosing coarse pixel a coverage of exactly 0.5.
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = mask[0, 1] = True
        coverage = reproject_coverage(
            mask,
            source_wcs=LinearWCS(scale=1.0),
            target_wcs=LinearWCS(scale=2.0, ra0=0.5, dec0=0.5),
            target_shape=(2, 2),
        )
        assert coverage[0, 0] == pytest.approx(0.5, rel=1e-6)


class TestApertureSEDResult:
    """Result export and plot helpers."""

    def test_to_table_and_plot_smoke(self) -> None:
        sed = _sed(
            {
                "F090W": _band(
                    np.ones((3, 3)),
                    wavelength=0.9,
                    wavelength_error=(0.05, 0.06),
                    flux_scale_mjy=0.01,
                    label_pixels={(1, 1)},
                )
            }
        )
        result = sed.draft(
            seed_xy_by_band={"F090W": (1.0, 1.0)},
            grow_kwargs=_grow_all_label_kwargs(),
        ).measure()

        table = result.to_table()
        assert table["band"][0] == "F090W"
        assert table["flux"][0] == pytest.approx(1.0)
        assert "flux_mjy" in table.colnames
        assert "covered_area" in table.colnames

        bokeh_fig = result.show(display_plot=False)
        assert bokeh_fig.title.text == "TEST aperture SED"
        assert bokeh_fig.yaxis[0].axis_label == "Flux (mJy)"
        assert bokeh_fig.xaxis[0].axis_label == "Wavelength (µm)"
        assert {r.glyph.__class__.__name__ for r in bokeh_fig.renderers} == {"Scatter"}
        assert bokeh_fig.renderers[0].glyph.y == "flux_mjy"

        mpl_fig = result.plot()
        assert mpl_fig.axes[0].get_title() == "TEST aperture SED"
        assert mpl_fig.axes[0].get_xlabel() == "Wavelength (µm)"
        assert mpl_fig.axes[0].get_ylabel() == "Flux (mJy)"


def _three_band_result() -> Any:
    """Measure a 3-band SED whose apertures all contain the seed pixel (3, 3)."""
    sed = _sed(
        {
            "F090W": _band(
                np.ones((7, 7)),
                wavelength=0.9,
                flux_scale_mjy=0.01,
                label_pixels=_block(range(3, 5), range(2, 5)),
                error=np.ones((7, 7)),
            ),
            "F150W": _band(
                np.ones((7, 7)),
                wavelength=1.5,
                flux_scale_mjy=0.01,
                label_pixels=_block(range(2, 4), range(3, 6)),
                error=np.ones((7, 7)),
            ),
            "F200W": _band(
                np.ones((7, 7)),
                wavelength=2.0,
                flux_scale_mjy=0.01,
                label_pixels=_block(range(3, 6), range(3, 5)),
                error=np.ones((7, 7)),
            ),
        },
        reference="F090W",
    )
    return sed.draft(
        seed_xy_by_band={k: (3.0, 3.0) for k in ("F090W", "F150W", "F200W")},
        grow_kwargs=_grow_all_label_kwargs(),
    ).measure()


class TestApertureBBox:
    """Footprint bounding box used to frame a zoomed-in thumbnail panel."""

    def test_tight_box_with_no_pad(self) -> None:
        fp = np.zeros((10, 10), dtype=bool)
        fp[4:6, 4:6] = True
        rsl, csl = _aperture_bbox(fp, pad=0.0)
        assert (rsl.start, rsl.stop) == (4, 6)
        assert (csl.start, csl.stop) == (4, 6)

    def test_pad_grows_box_by_extent_fraction(self) -> None:
        fp = np.zeros((10, 10), dtype=bool)
        fp[4:6, 4:6] = True  # 2x2 extent -> pad 0.5 grows by 1 each side.
        rsl, csl = _aperture_bbox(fp, pad=0.5)
        assert (rsl.start, rsl.stop) == (3, 7)
        assert (csl.start, csl.stop) == (3, 7)

    def test_pad_clips_to_array(self) -> None:
        fp = np.zeros((6, 6), dtype=bool)
        fp[0, 0] = fp[5, 5] = True
        rsl, csl = _aperture_bbox(fp, pad=1.0)
        assert (rsl.start, rsl.stop) == (0, 6)
        assert (csl.start, csl.stop) == (0, 6)

    def test_empty_footprint_returns_none(self) -> None:
        assert _aperture_bbox(np.zeros((4, 4), dtype=bool), pad=0.3) is None


class TestApertureThumbnails:
    """Self-contained matplotlib aperture montage and single-band thumbnail."""

    def test_result_keeps_native_band_images(self) -> None:
        result = _three_band_result()
        assert set(result.band_images) == {"F090W", "F150W", "F200W"}
        for name, image in result.band_images.items():
            assert image.shape == result.band_coverage[name].shape
            np.testing.assert_array_equal(image, np.ones((7, 7)))

    def test_montage_titles_one_panel_per_band(self) -> None:
        result = _three_band_result()
        fig = result.plot_apertures()
        titled = [ax for ax in fig.axes if ax.get_title()]
        assert len(titled) == 3
        # Default grid for 3 panels is 2 columns x 2 rows (one cell blank).
        assert fig.axes[0].get_subplotspec().get_geometry()[:2] == (2, 2)
        assert len(fig.axes) == 4

    def test_ncols_controls_grid(self) -> None:
        result = _three_band_result()
        fig = result.plot_apertures(ncols=3)
        assert fig.axes[0].get_subplotspec().get_geometry()[:2] == (1, 3)

    def test_unknown_band_raises(self) -> None:
        result = _three_band_result()
        with pytest.raises(ValueError, match="unknown band"):
            result.plot_apertures(bands=["F999W"])

    def test_coverage_overlay_toggle(self) -> None:
        result = _three_band_result()
        # Background image plus the union-coverage fill image when enabled.
        on = result.plot_thumbnail("F090W", show_coverage=True)
        assert len(on.axes[0].images) == 2
        off = result.plot_thumbnail("F090W", show_coverage=False)
        assert len(off.axes[0].images) == 1

    def test_seed_marker_toggle(self) -> None:
        result = _three_band_result()
        on = result.plot_thumbnail("F090W", show_seed=True)
        assert len(on.axes[0].lines) == 1
        off = result.plot_thumbnail("F090W", show_seed=False)
        assert len(off.axes[0].lines) == 0

    def test_full_cutout_by_default_zoom_in_frames_source(self) -> None:
        result = _three_band_result()
        # Default: the whole 7x7 cutout is shown (imshow spans -0.5..6.5 = 7 px).
        full = result.plot_thumbnail("F090W")
        full_x = full.axes[0].get_xlim()
        assert (full_x[1] - full_x[0]) >= 7.0
        # zoom_in frames the source, a strict sub-window of the full cutout.
        zoomed = result.plot_thumbnail("F090W", zoom_in=True)
        zoom_x = zoomed.axes[0].get_xlim()
        assert (zoom_x[1] - zoom_x[0]) < (full_x[1] - full_x[0])


class TestAllowBackground:
    """Default growth admits the background; ``allow_background=False`` confines."""

    def test_background_lets_growth_into_sky(self) -> None:
        data = np.ones((5, 5))
        labels = _label((5, 5), {(2, 2)})  # one source pixel, rest is background.

        confined = grow_aperture_mask(
            data,
            seed_xy=(2.0, 2.0),
            label_map=labels,
            allow_background=False,
            grow_kwargs=_grow_all_label_kwargs(),
        )
        spread = grow_aperture_mask(
            data,
            seed_xy=(2.0, 2.0),
            label_map=labels,
            allow_background=True,
            grow_kwargs=_grow_all_label_kwargs(),
        )

        # Confined: only the lone source pixel is admissible.
        assert int(confined.mask.sum()) == 1
        # Background allowed: growth leaves the segment into the sky.
        assert int(spread.mask.sum()) > 1
        assert bool((spread.mask & (labels == 0)).any())


class TestApertureFloor:
    """Area-scaled stop-check floor and its segment cap."""

    def test_generic_floor_finest_keeps_base(self) -> None:
        assert _generic_floor(scale=10.0, finest_scale=10.0, base=40) == 40

    def test_generic_floor_scales_by_area(self) -> None:
        # Half the linear scale (pixels/deg) -> a quarter of the sky area.
        assert _generic_floor(scale=5.0, finest_scale=10.0, base=40) == 10

    def test_generic_floor_clamped_at_lo(self) -> None:
        assert _generic_floor(scale=1.0, finest_scale=10.0, base=30) == 8

    def test_generic_floor_degenerate_scale_keeps_base(self) -> None:
        assert _generic_floor(scale=0.0, finest_scale=0.0, base=30) == 30

    def test_label_floor_caps_at_segment(self) -> None:
        labels = _label((6, 6), _block(range(2, 4), range(2, 4)))  # 4-pixel source
        data = np.ones((6, 6))
        assert _label_floor(30, labels, data, (2.0, 2.0)) == 4

    def test_label_floor_uses_generic_when_smaller(self) -> None:
        labels = _label((10, 10), _block(range(1, 9), range(1, 9)))  # 64-pixel source
        data = np.ones((10, 10))
        assert _label_floor(8, labels, data, (4.0, 4.0)) == 8

    def test_label_floor_background_seed_is_one(self) -> None:
        labels = _label((6, 6), {(2, 2)})
        data = np.ones((6, 6))
        # Seed (col 0, row 0) sits on the background label 0.
        assert _label_floor(30, labels, data, (0.0, 0.0)) == 1


class TestApertureSnr:
    """Background-subtracted detection SNR helper."""

    def test_no_background(self) -> None:
        assert aperture_snr(10.0, 2.0, None, 5.0) == pytest.approx(5.0)

    def test_with_background(self) -> None:
        assert aperture_snr(10.0, 2.0, 1.0, 4.0) == pytest.approx(3.0)

    def test_missing_error_is_none(self) -> None:
        assert aperture_snr(10.0, None, None, 5.0) is None

    def test_zero_error_is_none(self) -> None:
        assert aperture_snr(10.0, 0.0, None, 5.0) is None

    def test_nan_flux_is_none(self) -> None:
        assert aperture_snr(float("nan"), 2.0, None, 5.0) is None


class TestApertureNoise:
    """Correlation-aware error / background / SNR enrichment of a band."""

    def test_correlated_error_exceeds_formal(self) -> None:
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(5)
        corr = gaussian_filter(rng.normal(size=(64, 64)), 1.5)
        corr /= corr.std()  # unit per-pixel variance, but spatially correlated
        band = normalize_band(
            "t", _band(corr, error=np.ones((64, 64)), flux_scale_mjy=0.01)
        )
        cov = np.zeros((64, 64))
        cov[30:35, 30:35] = 1.0
        formal = measure_band(band, cov, flag_bad_fraction=0.1)

        enriched = measure_aperture_noise(
            band, cov, formal, max_lag=8, other_source_dilation=2
        )

        assert enriched.background_level is not None
        assert enriched.snr is not None
        assert enriched.error_uncorrelated == pytest.approx(formal.error_uncorrelated)
        # Positive spatial correlation inflates the aperture error over formal.
        assert enriched.error > formal.error_uncorrelated

    def test_too_little_sky_falls_back(self) -> None:
        rng = np.random.default_rng(6)
        band = normalize_band(
            "t", _band(rng.normal(size=(10, 10)), error=np.ones((10, 10)))
        )
        cov = np.zeros((10, 10))
        cov[4:6, 4:6] = 1.0
        formal = measure_band(band, cov, flag_bad_fraction=0.1)

        enriched = measure_aperture_noise(
            band, cov, formal, max_lag=8, other_source_dilation=2
        )

        assert enriched is formal  # returned untouched
        assert enriched.background_level is None
        assert enriched.error == formal.error_uncorrelated  # uncorrelated fallback


def _photometry(
    *,
    band: str,
    wavelength: float,
    flux_mjy: float,
    error_mjy: float | None,
    snr: float | None,
    flagged: bool = False,
) -> BandPhotometry:
    """Build a BandPhotometry for the SED-figure tests."""
    return BandPhotometry(
        band=band,
        wavelength=wavelength,
        wavelength_error=(0.05, 0.05),
        flux=flux_mjy * 100.0,
        error=None if error_mjy is None else error_mjy * 100.0,
        flux_mjy=flux_mjy,
        error_mjy=error_mjy,
        error_uncorrelated=None if error_mjy is None else error_mjy * 100.0,
        flux_scale_mjy=0.01,
        flux_unit=None,
        covered_area=10.0,
        valid_area=10.0,
        bad_fraction=0.0,
        background_level=None,
        snr=snr,
        flagged=flagged,
    )


class TestSedFigure:
    """Detection vs upper-limit rendering in the SED plot."""

    def test_splits_detections_and_upper_limits(self) -> None:
        ms = [
            _photometry(
                band="A", wavelength=1.0, flux_mjy=0.1, error_mjy=0.01, snr=10.0
            ),
            _photometry(
                band="B", wavelength=2.0, flux_mjy=0.001, error_mjy=0.01, snr=0.1
            ),
        ]
        fig = sed_figure(ms, detection_snr=2.0, upper_limit_sigma=2.0)
        markers = {r.glyph.marker for r in fig.renderers if hasattr(r.glyph, "marker")}
        assert "circle" in markers  # the detection
        assert "inverted_triangle" in markers  # the upper limit

    def test_unknown_snr_is_drawn_as_detection(self) -> None:
        ms = [
            _photometry(
                band="A", wavelength=1.0, flux_mjy=0.1, error_mjy=None, snr=None
            )
        ]
        fig = sed_figure(ms)
        markers = {r.glyph.marker for r in fig.renderers if hasattr(r.glyph, "marker")}
        assert markers == {"circle"}

    def test_rejects_low_detection_snr(self) -> None:
        ms = [
            _photometry(band="A", wavelength=1.0, flux_mjy=0.1, error_mjy=0.01, snr=5.0)
        ]
        with pytest.raises(ValueError, match="detection_snr"):
            sed_figure(ms, detection_snr=0.5)

    def test_rejects_nonpositive_upper_limit_sigma(self) -> None:
        ms = [
            _photometry(band="A", wavelength=1.0, flux_mjy=0.1, error_mjy=0.01, snr=5.0)
        ]
        with pytest.raises(ValueError, match="upper_limit_sigma"):
            sed_figure(ms, upper_limit_sigma=0.0)

    def test_mpl_renders_detections_and_limits(self) -> None:
        ms = [
            _photometry(
                band="A", wavelength=1.0, flux_mjy=0.1, error_mjy=0.01, snr=10.0
            ),
            _photometry(
                band="B", wavelength=2.0, flux_mjy=0.001, error_mjy=0.01, snr=0.1
            ),
        ]
        fig = sed_figure_mpl(ms, detection_snr=2.0, upper_limit_sigma=2.0)
        assert fig.axes[0].get_ylabel() == "Flux (mJy)"
        # one errorbar container for the detection, one for the upper limit
        assert len(fig.axes[0].containers) >= 2

    def test_mpl_rejects_low_detection_snr(self) -> None:
        ms = [
            _photometry(band="A", wavelength=1.0, flux_mjy=0.1, error_mjy=0.01, snr=5.0)
        ]
        with pytest.raises(ValueError, match="detection_snr"):
            sed_figure_mpl(ms, detection_snr=0.5)
