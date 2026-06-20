"""Persist built PSFs to multi-extension FITS and read them back.

A :class:`~noobfriend.extraction.psf.CorePsf` / :class:`EffectivePsf` is a small
bundle of arrays and scalars, so it is stored as one FITS file: the main PSF
plane as the primary image, the other planes as named image extensions, and the
scalars (oversample, stitch geometry, provenance counts) as header keywords. A
``NF_TYPE`` keyword tags the file so the wrong loader fails loudly.

The hybrid is stored as its own arrays rather than as the noobase ``ExtendedPsf``
handle, because re-stitching the already-baked wing on load would re-process it.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from noobfriend.extraction.psf._build import CorePsf, EffectivePsf

#: FITS layout version stamped into the ``NF_VER`` header keyword.
_PSF_FORMAT_VERSION: int = 1


def _prepare_path(path: str | Path, overwrite: bool) -> Path:
    """Validate the destination, refusing to clobber unless ``overwrite``."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it."
        )
    return path


def _require_type(header: object, expected: str) -> None:
    """Raise if the file's ``NF_TYPE`` is not the expected PSF kind."""
    got = header.get("NF_TYPE")  # type: ignore[attr-defined]
    if got != expected:
        raise ValueError(f"expected a {expected} FITS file; got NF_TYPE={got!r}.")


def save_core(psf: "CorePsf", path: str | Path, *, overwrite: bool = False) -> None:
    """Write a :class:`CorePsf` to a multi-extension FITS file."""
    from astropy.io import fits

    path = _prepare_path(path, overwrite)
    primary = fits.PrimaryHDU(data=np.asarray(psf.core, dtype=np.float64))
    header = primary.header
    header["NF_TYPE"] = "CorePsf"
    header["NF_VER"] = _PSF_FORMAT_VERSION
    header["OVERSAMP"] = psf.oversample
    header["CORESIZE"] = psf.core_size
    header["CONVERGD"] = psf.converged
    header["NSTARS"] = psf.n_stars
    header["NSOURCE"] = psf.n_sources
    fits.HDUList(
        [
            primary,
            fits.ImageHDU(
                np.asarray(psf.phase_grid, dtype=np.int64), name="PHASE_GRID"
            ),
            fits.ImageHDU(np.asarray(psf.flux, dtype=np.float64), name="FLUX"),
        ]
    ).writeto(path, overwrite=True)


def load_core(path: str | Path) -> "CorePsf":
    """Read a :class:`CorePsf` from a file written by :func:`save_core`."""
    from astropy.io import fits

    from noobfriend.extraction.psf._build import CorePsf

    with fits.open(path) as hdul:
        header = hdul[0].header
        _require_type(header, "CorePsf")
        return CorePsf(
            core=np.asarray(hdul[0].data, dtype=float),
            oversample=int(header["OVERSAMP"]),
            core_size=int(header["CORESIZE"]),
            flux=np.asarray(hdul["FLUX"].data, dtype=float),
            converged=bool(header["CONVERGD"]),
            n_stars=int(header["NSTARS"]),
            n_sources=int(header["NSOURCE"]),
            phase_grid=np.asarray(hdul["PHASE_GRID"].data, dtype=int),
        )


def save_effective(
    psf: "EffectivePsf", path: str | Path, *, overwrite: bool = False
) -> None:
    """Write an :class:`EffectivePsf` (and its nested core) to one FITS file."""
    from astropy.io import fits

    path = _prepare_path(path, overwrite)
    primary = fits.PrimaryHDU(data=np.asarray(psf.core_plane, dtype=np.float64))
    header = primary.header
    header["NF_TYPE"] = "EffectivePsf"
    header["NF_VER"] = _PSF_FORMAT_VERSION
    header["OVERSAMP"] = psf.oversample
    header["MATCHRAD"] = psf.match_radius
    header["FEATHERW"] = psf.feather_width
    header["EEAPRAD"] = psf.ee_aperture_radius
    header["WINGSIZE"] = psf.wing_size
    header["NWINGSTR"] = psf.n_wing_stars

    core = psf.core
    core_hdu = fits.ImageHDU(np.asarray(core.core, dtype=np.float64), name="CORE")
    core_header = core_hdu.header
    core_header["CORESIZE"] = core.core_size
    core_header["CONVERGD"] = core.converged
    core_header["NSTARS"] = core.n_stars
    core_header["NSOURCE"] = core.n_sources

    fits.HDUList(
        [
            primary,
            fits.ImageHDU(np.asarray(psf.wing, dtype=np.float64), name="WING"),
            core_hdu,
            fits.ImageHDU(
                np.asarray(core.phase_grid, dtype=np.int64), name="CORE_PHASE_GRID"
            ),
            fits.ImageHDU(np.asarray(core.flux, dtype=np.float64), name="CORE_FLUX"),
        ]
    ).writeto(path, overwrite=True)


def load_effective(path: str | Path) -> "EffectivePsf":
    """Read an :class:`EffectivePsf` from a file written by :func:`save_effective`."""
    from astropy.io import fits

    from noobfriend.extraction.psf._build import CorePsf, EffectivePsf

    with fits.open(path) as hdul:
        header = hdul[0].header
        _require_type(header, "EffectivePsf")
        core_header = hdul["CORE"].header
        core = CorePsf(
            core=np.asarray(hdul["CORE"].data, dtype=float),
            oversample=int(header["OVERSAMP"]),
            core_size=int(core_header["CORESIZE"]),
            flux=np.asarray(hdul["CORE_FLUX"].data, dtype=float),
            converged=bool(core_header["CONVERGD"]),
            n_stars=int(core_header["NSTARS"]),
            n_sources=int(core_header["NSOURCE"]),
            phase_grid=np.asarray(hdul["CORE_PHASE_GRID"].data, dtype=int),
        )
        return EffectivePsf(
            core=core,
            core_plane=np.asarray(hdul[0].data, dtype=float),
            wing=np.asarray(hdul["WING"].data, dtype=float),
            oversample=int(header["OVERSAMP"]),
            match_radius=float(header["MATCHRAD"]),
            feather_width=float(header["FEATHERW"]),
            ee_aperture_radius=float(header["EEAPRAD"]),
            wing_size=int(header["WINGSIZE"]),
            n_wing_stars=int(header["NWINGSTR"]),
        )
