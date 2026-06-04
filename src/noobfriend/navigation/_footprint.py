"""The :class:`Footprint`: a serializable on-sky region for a JWST product."""

from typing import Self

from gwcs import WCS
from pydantic import BaseModel

from noobfriend.extraction._wcs import world_detector_transforms


class Footprint(BaseModel):
    """The on-sky region a product covers, as world-space corner vertices.

    A footprint is the small, serializable summary of a product's astrometry:
    the ``(ra, dec)`` corners of its data array, in degrees. It is the field a
    manifest persists in place of the full (unserializable) GWCS, and a
    NooBook carries ``None`` for it when the product's WCS is not yet assigned.

    Attributes
    ----------
    corners : list of tuple of float
        World ``(ra, dec)`` vertices in degrees, ordered around the array
        boundary. Typically the four detector corners.
    """

    corners: list[tuple[float, float]]

    @classmethod
    def from_corners(cls, corners: list[tuple[float, float]]) -> Self:
        """Build a footprint from explicit ``(ra, dec)`` vertices in degrees."""
        return cls(corners=corners)

    @classmethod
    def from_wcs(cls, wcs: WCS, shape: tuple[int, ...]) -> Self:
        """Build a footprint from a GWCS and the science array shape.

        Parameters
        ----------
        wcs : gwcs.WCS
            The product's world-coordinate system.
        shape : tuple of int
            The science array shape in NumPy order; the last two entries are
            read as ``(ny, nx)``, so 2-D frames and 3-D cubes are both handled.

        Returns
        -------
        Footprint
            The four detector corners mapped to world coordinates.

        Notes
        -----
        Corners are mapped through the undispersed ``detector -> world``
        astrometry (the same transform cutouts use), so the footprint of a WFSS
        grism exposure is its direct-image field, not the dispersed trace.
        """
        ny, nx = shape[-2], shape[-1]
        _, detector_to_world = world_detector_transforms(wcs)
        pixel_corners = [(0, 0), (nx, 0), (nx, ny), (0, ny)]
        corners = [
            (float(ra), float(dec))
            for ra, dec in (detector_to_world(x, y) for x, y in pixel_corners)
        ]
        return cls(corners=corners)
