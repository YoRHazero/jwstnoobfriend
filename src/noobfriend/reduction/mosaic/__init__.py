"""Stage-3 mosaic-combination helpers (geometry and catalogs; no jwst / FITS).

Output-grid tiling (footprint corners in / grid + tile geometry out), the
astrometric reference pipeline (GAIA query, cleaning, image-source selection and
tweakreg-catalog formatting) that stage-3 resampling and alignment consume, the
deep-self-catalog alignment that replaces jwst's ``TweakRegStep``, and the
field-median outlier flagging that replaces jwst's imaging
``OutlierDetectionStep``.
"""

from noobfriend.reduction.mosaic._align import FrameSources, align_group
from noobfriend.reduction.mosaic._coadd import CoaddTile, TileCoadd, noise_kernel
from noobfriend.reduction.mosaic._astrometry import (
    GaiaReference,
    ReferenceProvider,
    build_reference,
    clean_gaia,
    query_gaia,
    select_point_sources,
    to_tweakreg_catalog,
    within_footprints,
)
from noobfriend.reduction.mosaic._outlier import (
    OUTLIER_DQ,
    FieldMedian,
    blot_to_frame,
    flag_outliers,
)
from noobfriend.reduction.mosaic._sky import SkyMatcher, frame_sky
from noobfriend.reduction.mosaic._tiling import (
    FieldGrid,
    TileSpec,
    field_grid,
    tile_grid,
    tile_gwcs,
    tile_members,
    tile_resample_params,
)

__all__ = [
    "OUTLIER_DQ",
    "CoaddTile",
    "FieldGrid",
    "FieldMedian",
    "FrameSources",
    "GaiaReference",
    "ReferenceProvider",
    "SkyMatcher",
    "TileCoadd",
    "TileSpec",
    "align_group",
    "blot_to_frame",
    "build_reference",
    "clean_gaia",
    "field_grid",
    "flag_outliers",
    "frame_sky",
    "noise_kernel",
    "query_gaia",
    "select_point_sources",
    "tile_grid",
    "tile_gwcs",
    "tile_members",
    "tile_resample_params",
    "to_tweakreg_catalog",
    "within_footprints",
]
