"""Tests for the noob 3a output: tile GWCS + asdf-in-FITS write / read-back.

Synthetic only: a coadd-like tile is written with an embedded GWCS and read back
through the same reader stack (NooBook) to confirm the jwst-free ASDF-in-FITS
product is downstream-compatible.
"""

import numpy as np
from astropy.io import fits

from noobfriend.core.io import write_asdf_fits
from noobfriend.navigation import NooBook
from noobfriend.reduction.mosaic import TileSpec, field_grid, tile_gwcs

from ._helpers import corners, tan_wcs

RA0, DEC0 = 150.0, 2.0
SCALE = 0.05


def _tan(shape):
    return tan_wcs(RA0, DEC0, shape, SCALE)


def _field(shape=(200, 200)):
    return field_grid([corners(_tan(shape), shape)], SCALE, rotation=0.0)


def test_tile_gwcs_matches_field_wcs():
    field = _field()
    tile = TileSpec(1, 2, 40, 30, 60, 50)  # an offset sub-window
    gwcs = tile_gwcs(field, tile)
    for tx, ty in [(0, 0), (25, 20), (59, 49)]:
        got = np.asarray(gwcs(tx, ty), dtype=float)
        want = field.wcs.pixel_to_world_values(tx + tile.x0, ty + tile.y0)
        assert np.allclose(got, np.asarray(want, float), atol=1e-10)


def test_tile_gwcs_inverts():
    field = _field()
    tile = TileSpec(0, 0, 0, 0, field.shape[1], field.shape[0])
    gwcs = tile_gwcs(field, tile)
    ra, dec = gwcs(70.0, 90.0)
    x, y = gwcs.invert(float(ra), float(dec))
    assert abs(x - 70.0) < 1e-6 and abs(y - 90.0) < 1e-6


def test_write_asdf_fits_roundtrips_through_noobook(tmp_path):
    field = _field()
    tile = TileSpec(0, 0, 0, 0, 80, 80)
    shape = (tile.ny, tile.nx)
    rng = np.random.default_rng(0)
    planes = {
        "SCI": rng.standard_normal(shape).astype("f4"),
        "ERR": np.ones(shape, "f4"),
        "WHT": np.full(shape, 4.0, "f4"),
        "NOISEKERN": rng.standard_normal((17, 17)).astype("f8"),
    }
    tree = {
        "meta": {
            "wcs": tile_gwcs(field, tile),
            "instrument": {"pupil": "CLEAR", "filter": "F444W"},
            "noise": {"max_lag": 8},
        }
    }
    raw = write_asdf_fits(planes, tree)
    # write the synthetic product to disk so every read path (header + data +
    # wcs) goes through the file, exercising the full downstream reader
    path = tmp_path / "jw01895001001_obs_t00x00_s0p050_3a.fits"
    path.write_bytes(raw)

    book = NooBook.from_file(str(path), "3a")
    assert book.pupil == "CLEAR"
    assert book.filter == "F444W"
    assert book.shape == shape
    ra, dec = book.wcs(39.5, 39.5)
    assert abs(float(ra) - RA0) < 1e-3 and abs(float(dec) - DEC0) < 1e-3
    assert np.allclose(np.asarray(book.data), planes["SCI"], atol=1e-6)


def test_write_asdf_fits_keeps_all_planes_and_trailing_asdf():
    field = _field()
    tile = TileSpec(0, 0, 0, 0, 32, 32)
    planes = {
        "SCI": np.zeros((32, 32), "f4"),
        "ERR": np.ones((32, 32), "f4"),
        "WHT": np.ones((32, 32), "f4"),
        "NOISEKERN": np.arange(17 * 17, dtype="f8").reshape(17, 17),
    }
    tree = {"meta": {"wcs": tile_gwcs(field, tile), "instrument": {"pupil": "CLEAR"}}}
    raw = write_asdf_fits(planes, tree)

    from io import BytesIO

    with fits.open(BytesIO(raw)) as hdul:
        names = [hdu.name for hdu in hdul]
        assert names[:5] == ["PRIMARY", "SCI", "ERR", "WHT", "NOISEKERN"]
        assert names[-1] == "ASDF"  # trailing ASDF for the tail reader
        assert hdul["NOISEKERN"].data.shape == (17, 17)
