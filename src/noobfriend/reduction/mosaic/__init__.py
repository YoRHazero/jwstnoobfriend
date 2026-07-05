"""Stage-3 mosaic-combination helpers (geometry and catalogs; no jwst / FITS).

Output-grid tiling (footprint corners in / grid + tile geometry out) and the
astrometric reference pipeline (GAIA query, cleaning, image-source selection and
tweakreg-catalog formatting) that stage-3 resampling and alignment consume.
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
from noobfriend.reduction.mosaic._tiling import (
    FieldGrid,
    TileSpec,
    field_grid,
    tile_grid,
    tile_members,
    tile_resample_params,
)

__all__ = [
    "FieldGrid",
    "GaiaReference",
    "ReferenceProvider",
    "TileSpec",
    "build_reference",
    "clean_gaia",
    "field_grid",
    "query_gaia",
    "select_point_sources",
    "tile_grid",
    "tile_members",
    "tile_resample_params",
    "to_tweakreg_catalog",
    "within_footprints",
]
