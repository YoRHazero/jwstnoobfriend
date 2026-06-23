"""Tests for the stage-3 pure helpers: output tiling and astrometry inputs.

Logic only -- no FITS, no network. Tiling is exercised on synthetic footprints
and a hand-built grid; astrometry on a synthetic GAIA table, a synthetic source
catalogue and a fake reference provider.
"""

import sys
import time
import types

import numpy as np
import pytest
from astropy.table import Table

from noobfriend.core.imgutils import SourceCatalog
from noobfriend.reduction import (
    FieldGrid,
    build_reference,
    clean_gaia,
    field_grid,
    query_gaia,
    select_point_sources,
    tile_grid,
    tile_members,
    tile_resample_params,
    to_tweakreg_catalog,
    within_footprints,
)


def _square(ra0: float, dec0: float, half_deg: float) -> np.ndarray:
    """Return a 4-corner square footprint centred on (ra0, dec0)."""
    return np.array(
        [
            [ra0 - half_deg, dec0 - half_deg],
            [ra0 + half_deg, dec0 - half_deg],
            [ra0 + half_deg, dec0 + half_deg],
            [ra0 - half_deg, dec0 + half_deg],
        ]
    )


# -- field_grid ---------------------------------------------------------------


def test_field_grid_contains_every_corner() -> None:
    fp = _square(53.1, -27.75, 0.02)
    field = field_grid([fp], pixel_scale=0.04)
    x, y = field.wcs.world_to_pixel_values(fp[:, 0], fp[:, 1])
    assert x.min() >= 0 and y.min() >= 0
    assert x.max() <= field.shape[1] and y.max() <= field.shape[0]
    # side 0.04deg=144"/0.04"=3600 px in Dec; RA compresses by cos(dec)~0.885
    assert 3500 < field.shape[0] < 3700  # ny (Dec)
    assert 3000 < field.shape[1] < 3400  # nx (RA, compressed)


def test_field_grid_centres_on_the_footprints() -> None:
    field = field_grid([_square(189.2, 62.25, 0.01)], pixel_scale=0.04)
    assert field.crval[0] == pytest.approx(189.2, abs=1e-6)
    assert field.crval[1] == pytest.approx(62.25, abs=1e-6)


def _rotated_strip(angle_deg: float) -> list[np.ndarray]:
    """Return a 3x1 strip of square footprints rolled by ``angle_deg``."""
    base = []
    for k in range(3):  # three abutting squares along the (pre-rotation) x-axis
        cx = (k - 1) * 0.04
        base.append(
            np.array(
                [
                    [cx - 0.02, -0.02],
                    [cx + 0.02, -0.02],
                    [cx + 0.02, 0.02],
                    [cx - 0.02, 0.02],
                ]
            )
        )
    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cdec = 0.0
    out = []
    for sq in base:
        xy = sq.copy()
        xy[:, 0] *= np.cos(np.deg2rad(cdec))  # to a local tangent frame
        xy = xy @ rot.T
        xy[:, 0] = 60.0 + xy[:, 0] / np.cos(np.deg2rad(cdec))
        xy[:, 1] = cdec + xy[:, 1]
        out.append(xy)
    return out


def test_field_grid_auto_rotation_beats_north_up_on_a_rolled_field() -> None:
    strip = _rotated_strip(40.0)  # an elongated field rolled 40 deg from north
    auto = field_grid(strip, pixel_scale=0.05)  # rotation="auto" by default
    north = field_grid(strip, pixel_scale=0.05, rotation=0.0)
    auto_area = auto.shape[0] * auto.shape[1]
    north_area = north.shape[0] * north.shape[1]
    assert auto_area < 0.7 * north_area  # roll-aligned packs the strip much tighter
    assert 25.0 < (auto.rotation % 90.0) < 55.0  # recovered ~the 40 deg roll


def test_field_grid_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least one footprint"):
        field_grid([], pixel_scale=0.04)
    with pytest.raises(ValueError, match="pixel_scale must be positive"):
        field_grid([_square(0, 0, 0.01)], pixel_scale=0.0)


# -- tile_grid ----------------------------------------------------------------


def _grid(ny: int, nx: int) -> FieldGrid:
    """Return a FieldGrid carrying only the shape the tiling math needs."""
    return FieldGrid(
        wcs=None,
        shape=(ny, nx),
        crval=(0.0, 0.0),
        crpix=(1.0, 1.0),
        pixel_scale=0.04,
        rotation=0.0,
    )


def test_tile_grid_even_split_has_no_sliver() -> None:
    tiles = tile_grid(_grid(10000, 9000), target_size=4096, overlap=128)
    # 9000 -> ceil(9000/4096)=3 cols; 10000 -> 3 rows; 3x3 tiles
    assert {(t.iy, t.ix) for t in tiles} == {(i, j) for i in range(3) for j in range(3)}
    # cores ~3000 / ~3334, so no tile is a thin remainder
    widths = sorted({t.nx for t in tiles})
    assert min(widths) > 2500


def test_tile_grid_core_sizes_differ_by_at_most_one() -> None:
    # 10000 / 3 tiles -> cores 3334, 3333, 3333 (remainder spread, no short last)
    tiles = tile_grid(_grid(10000, 9000), target_size=4096, overlap=0)
    col0 = sorted(t.ny for t in tiles if t.ix == 0)
    assert max(col0) - min(col0) <= 1
    assert sum(t.ny for t in tiles if t.ix == 0) == 10000  # exact cover, no overlap


def test_tile_grid_tiles_cover_the_whole_field() -> None:
    ny, nx = 10000, 9000
    tiles = tile_grid(_grid(ny, nx), target_size=4096, overlap=128)
    top_left = [t for t in tiles if t.ix == 0 and t.iy == 0][0]
    bot_right = max(tiles, key=lambda t: (t.iy, t.ix))
    assert top_left.x0 == 0 and top_left.y0 == 0
    assert bot_right.x0 + bot_right.nx == nx
    assert bot_right.y0 + bot_right.ny == ny


def test_tile_grid_interior_tiles_carry_overlap() -> None:
    tiles = tile_grid(_grid(10000, 9000), target_size=4096, overlap=128)
    # an interior tile starts one overlap-width before its core boundary
    interior = [t for t in tiles if t.ix == 1][0]
    assert interior.x0 == pytest.approx(3000 - 128, abs=1)


def test_tile_grid_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="target_size must be positive"):
        tile_grid(_grid(100, 100), target_size=0)
    with pytest.raises(ValueError, match="overlap must be non-negative"):
        tile_grid(_grid(100, 100), overlap=-1)


# -- tile_members / coverage --------------------------------------------------


def test_tile_members_includes_overlapping_drops_distant() -> None:
    near = _square(53.10, -27.75, 0.02)
    far = _square(53.50, -27.75, 0.02)  # well outside the near field
    field = field_grid([near], pixel_scale=0.04)
    whole = tile_grid(field, target_size=100000)[0]  # one tile = the field
    members, coverage = tile_members(field, whole, [near, far])
    assert members == [0]
    assert coverage > 0.5  # the near square fills most of its own field


def test_tile_members_empty_when_nothing_overlaps() -> None:
    field = field_grid([_square(53.1, -27.75, 0.02)], pixel_scale=0.04)
    whole = tile_grid(field, target_size=100000)[0]
    members, coverage = tile_members(field, whole, [_square(10.0, 10.0, 0.02)])
    assert members == []
    assert coverage == 0.0


def test_tile_resample_params_share_projection_shift_crpix() -> None:
    field = field_grid([_square(53.1, -27.75, 0.03)], pixel_scale=0.04)
    tiles = tile_grid(field, target_size=2048, overlap=64)
    p0 = tile_resample_params(field, tiles[0])
    p1 = tile_resample_params(field, tiles[-1])
    assert p0["crval"] == p1["crval"]  # one projection for the whole mosaic
    assert p0["pixel_scale"] == field.pixel_scale
    assert p0["output_shape"] == [tiles[0].nx, tiles[0].ny]
    # crpix shifts by the tile origin, converted from 1-based (FITS) to 0-based (jwst)
    assert p1["crpix"][0] == pytest.approx(field.crpix[0] - 1 - tiles[-1].x0)


# -- clean_gaia ---------------------------------------------------------------


def _gaia_table() -> Table:
    return Table(
        {
            "ra": [10.0, 10.1, 10.2, 10.3],
            "dec": [0.0, 0.0, 0.0, 0.0],
            "pmra": [1000.0, 0.0, 0.0, 0.0],  # 1"/yr (1000 mas/yr) for the first
            "pmdec": [0.0, 0.0, 0.0, 0.0],
            "ref_epoch": [2016.0, 2016.0, 2016.0, 2016.0],
            "ruwe": [1.0, 2.0, 1.0, 1.0],  # second fails ruwe
            "astrometric_excess_noise": [0.0, 0.0, 5.0, 0.0],  # third fails noise
            "classprob_dsc_combmod_galaxy": [0.0, 0.0, 0.0, 0.9],  # fourth is a galaxy
        }
    )


def test_clean_gaia_propagates_proper_motion() -> None:
    out = clean_gaia(_gaia_table(), obs_epoch=2026.0)
    # only source 0 survives the quality cuts; 10 yr * 1"/yr / cos(0) = 10" in RA
    assert len(out) == 1
    assert out["ra"][0] == pytest.approx(10.0 + 10 / 3600, abs=1e-6)


def test_clean_gaia_drops_blends_and_galaxies() -> None:
    out = clean_gaia(_gaia_table(), obs_epoch=2016.0)
    assert len(out) == 1  # the high-ruwe, high-noise and galaxy rows are gone


# -- source selection / catalog formatting ------------------------------------


def _source_catalog() -> SourceCatalog:
    n = 4
    z = np.zeros(n)
    return SourceCatalog(
        x=np.array([1.0, 2.0, 3.0, 4.0]),
        y=np.array([1.0, 2.0, 3.0, 4.0]),
        ra=z.copy(),
        dec=z.copy(),
        fwhm=np.array([2.0, 2.0, 8.0, 2.0]),  # 3rd too big (extended)
        ellipticity=np.array([0.05, 0.5, 0.05, 0.05]),  # 2nd elongated (blend)
        flux=np.ones(n),
        peak=np.ones(n),
        snr=np.array([100.0, 100.0, 100.0, 5.0]),  # 4th too faint
        sharpness=np.full(n, 2.0),
        nn_dist=np.full(n, 50.0),
    )


def test_select_point_sources_keeps_only_clean_isolated_psf() -> None:
    stars = select_point_sources(_source_catalog())
    assert len(stars) == 1
    assert stars.x[0] == 1.0


def test_to_tweakreg_catalog_has_xy_columns() -> None:
    table = to_tweakreg_catalog(_source_catalog())
    assert set(table.colnames) == {"x", "y"}
    assert len(table) == 4


# -- reference assembly (no network) ------------------------------------------


def test_build_reference_stacks_providers() -> None:
    def fake(ra: float, dec: float, radius: float, epoch: float) -> Table:
        return Table({"RA": [ra, ra + 1], "DEC": [dec, dec + 1]})

    out = build_reference(10.0, 20.0, 0.1, 2023.0, providers=(fake, fake))
    assert set(out.colnames) == {"RA", "DEC"}
    assert len(out) == 4


def test_within_footprints_keeps_inside_drops_outside() -> None:
    fp = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]  # unit square
    ra = np.array([0.5, 2.0, 0.1, -0.1])
    dec = np.array([0.5, 2.0, 0.9, 0.5])
    mask = within_footprints(ra, dec, fp)
    assert list(mask) == [True, False, True, False]


def test_build_reference_trims_to_footprints() -> None:
    def fake(ra: float, dec: float, radius: float, epoch: float) -> Table:
        # one source inside the unit square, one far outside (cone over-query)
        return Table({"RA": [0.5, 5.0], "DEC": [0.5, 5.0]})

    fp = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]
    out = build_reference(0.0, 0.0, 1.0, 2023.0, providers=(fake,), footprints=fp)
    assert len(out) == 1
    assert out["RA"][0] == 0.5


# -- query_gaia retry on transient failures (fake astroquery, no network) -----


def _install_fake_gaia(
    monkeypatch: pytest.MonkeyPatch, side_effects: list[object]
) -> list[str]:
    """Stub ``astroquery.gaia`` so ``query_gaia`` never touches the network.

    Each entry of ``side_effects`` is consumed by one ``launch_job_async`` call:
    an :class:`Exception` is raised, anything else is returned as the job's
    ``get_results()``. Returns the list of ADQL strings seen, one per call.
    """
    seq = list(side_effects)
    calls: list[str] = []

    class _FakeJob:
        def __init__(self, result: object) -> None:
            self._result = result

        def get_results(self) -> object:
            return self._result

    class _FakeGaia:
        ROW_LIMIT: int = -1

        @staticmethod
        def launch_job_async(adql: str) -> "_FakeJob":
            calls.append(adql)
            effect = seq.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return _FakeJob(effect)

    astroquery_mod = types.ModuleType("astroquery")
    gaia_mod = types.ModuleType("astroquery.gaia")
    gaia_mod.Gaia = _FakeGaia  # type: ignore[attr-defined]
    astroquery_mod.gaia = gaia_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "astroquery", astroquery_mod)
    monkeypatch.setitem(sys.modules, "astroquery.gaia", gaia_mod)
    return calls


def test_query_gaia_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = Table({"ra": [1.0], "dec": [2.0]})
    calls = _install_fake_gaia(
        monkeypatch,
        [
            ConnectionResetError(104, "reset"),
            ConnectionResetError(104, "reset"),
            sentinel,
        ],
    )
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    out = query_gaia(53.1, -27.8, 0.1, retries=5, backoff=1.0)

    assert out is sentinel
    assert len(calls) == 3  # two failures, then success
    assert sleeps == [1.0, 2.0]  # exponential backoff between attempts


def test_query_gaia_reraises_after_exhausting(monkeypatch: pytest.MonkeyPatch) -> None:
    err = ConnectionResetError(104, "reset")
    calls = _install_fake_gaia(monkeypatch, [err, err, err])
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(ConnectionResetError):
        query_gaia(53.1, -27.8, 0.1, retries=3, backoff=1.0)

    assert len(calls) == 3  # all attempts used
    assert sleeps == [1.0, 2.0]  # no sleep after the final failure


def test_query_gaia_rejects_nonpositive_retries() -> None:
    with pytest.raises(ValueError):
        query_gaia(53.1, -27.8, 0.1, retries=0)
