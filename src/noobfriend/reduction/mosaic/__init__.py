"""Stage-3 mosaic-combination helpers (geometry and catalogs; no jwst / FITS).

Output-grid tiling (footprint corners in / grid + tile geometry out), the
astrometric reference pipeline (GAIA query, cleaning, image-source selection and
tweakreg-catalog formatting) that stage-3 resampling and alignment consume, and
the field-median outlier flagging that replaces jwst's imaging
``OutlierDetectionStep``.
"""

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
from noobfriend.reduction.mosaic._tiling import (
    FieldGrid,
    TileSpec,
    field_grid,
    tile_grid,
    tile_members,
    tile_resample_params,
)

__all__ = [
    "OUTLIER_DQ",
    "FieldGrid",
    "FieldMedian",
    "GaiaReference",
    "ReferenceProvider",
    "TileSpec",
    "blot_to_frame",
    "build_reference",
    "clean_gaia",
    "field_grid",
    "flag_outliers",
    "query_gaia",
    "select_point_sources",
    "tile_grid",
    "tile_members",
    "tile_resample_params",
    "to_tweakreg_catalog",
    "within_footprints",
]
