"""Parse JWST product file names into their structured fields.

Exposure-level products follow the fixed pattern
``jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>_<detector>_<suffix>.fits``; stage-3
association products instead start ``jw<ppppp>-<acid>_...``. The helpers here
turn a file name into the subset of identifiers readable from the name alone,
with no file access.
"""

import re
from dataclasses import dataclass

_FITS_SUFFIXES: tuple[str, ...] = (".fits.gz", ".fits")

_EXPOSURE_RE: re.Pattern[str] = re.compile(
    r"^jw"
    r"(?P<program>\d{5})"
    r"(?P<observation>\d{3})"
    r"(?P<visit>\d{3})"
    r"_(?P<ggsaa>[a-z0-9]{5})"
    r"_(?P<exposure>\d{5})"
    r"_(?P<detector>[a-z0-9]+)"
    r"_(?P<suffix>.+)$"
)

_STAGE3_RE: re.Pattern[str] = re.compile(r"^jw(?P<program>\d{5})-[a-z0-9]+_")


@dataclass(frozen=True)
class ExposureName:
    """The fields parsed from an exposure-level JWST product name.

    Attributes
    ----------
    program_id : str
        Five-digit program (proposal) identifier.
    observation, visit : str
        Three-digit observation and visit numbers.
    ggsaa : str
        The five-character ``<visit_group><parallel_seq><activity>`` token.
    exposure : str
        Five-digit exposure number.
    detector : str
        Detector name, e.g. ``"nrca1"``.
    suffix : str
        Product suffix, e.g. ``"rate"``, ``"cal"``, ``"i2d"``.
    """

    program_id: str
    observation: str
    visit: str
    ggsaa: str
    exposure: str
    detector: str
    suffix: str


def strip_fits_suffix(filename: str) -> str:
    """Return ``filename`` without a trailing ``.fits`` / ``.fits.gz``."""
    for suffix in _FITS_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def parse_exposure_name(filename: str) -> ExposureName | None:
    """Parse an exposure-level product name, or return ``None``.

    Parameters
    ----------
    filename : str
        A product file name or stem (any ``.fits`` extension is ignored).

    Returns
    -------
    ExposureName or None
        The parsed fields, or ``None`` when ``filename`` is not an
        exposure-level name (e.g. a stage-3 association product).
    """
    match = _EXPOSURE_RE.match(strip_fits_suffix(filename))
    if match is None:
        return None
    return ExposureName(
        program_id=match["program"],
        observation=match["observation"],
        visit=match["visit"],
        ggsaa=match["ggsaa"],
        exposure=match["exposure"],
        detector=match["detector"],
        suffix=match["suffix"],
    )


def parse_program_id(filename: str) -> str | None:
    """Return the five-digit program id from any JWST product name.

    Works for both exposure-level (``jw01345001001_...``) and stage-3
    association (``jw01345-o001_...``) names.

    Parameters
    ----------
    filename : str
        A product file name or stem.

    Returns
    -------
    str or None
        The program id, or ``None`` if it cannot be read from the name.
    """
    stem = strip_fits_suffix(filename)
    exposure = _EXPOSURE_RE.match(stem)
    if exposure is not None:
        return exposure["program"]
    stage3 = _STAGE3_RE.match(stem)
    if stage3 is not None:
        return stage3["program"]
    return None
