"""The user-facing :class:`Cutout`.

A :class:`Cutout` couples a target :class:`gwcs.WCS` with a resolved
:class:`~noobfriend.extraction.cutout._geometry.CutoutGeometry`. It is built
through one of three explicit factory methods — :meth:`Cutout.from_pixel`,
:meth:`Cutout.from_world`, :meth:`Cutout.from_bounds` — so every construction
path is self-documenting and no invalid combination of inputs is representable.

Once built, a cutout is immutable. It can report its world-space meshgrid and
footprint, and extract the region from any data array two ways:

- :meth:`Cutout.array` — an axis-aligned integer slice (fast, no resampling).
- :meth:`Cutout.reproject` — a flux-conserving reprojection onto the cutout
  grid via ``noobase`` (for data on a *different* WCS).

``Cutout`` is the only public name in this subpackage; everything it builds on
(``_geometry``, ``_slice``, ``_reproject`` and the package-level ``_wcs``) is
internal.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
from astropy import units as u
from gwcs import WCS

from noobfriend.extraction._wcs import (
    pixel_scale_per_deg,
    world_detector_transforms,
)
from noobfriend.extraction.cutout._geometry import (
    CutoutGeometry,
    resolve_offsets,
    round_index,
)
from noobfriend.extraction.cutout._reproject import reproject_onto
from noobfriend.extraction.cutout._slice import slice_with_fill


@dataclass(frozen=True)
class Cutout:
    """A cutout region resolved against a target WCS.

    Construct via :meth:`from_pixel`, :meth:`from_world`, or :meth:`from_bounds`
    rather than calling the constructor directly.

    Attributes
    ----------
    wcs : gwcs.WCS
        The WCS the cutout's pixel bounds are defined against.
    geometry : CutoutGeometry
        The resolved half-open integer pixel bounds.
    """

    wcs: WCS
    geometry: CutoutGeometry

    @classmethod
    def from_bounds(
        cls, wcs: WCS, *, x_bounds: tuple[int, int], y_bounds: tuple[int, int]
    ) -> Self:
        """Build a cutout from explicit half-open pixel bounds.

        Parameters
        ----------
        wcs : gwcs.WCS
            The target WCS.
        x_bounds, y_bounds : tuple[int, int]
            Half-open ``(start, end)`` column / row bounds.

        Returns
        -------
        Cutout
        """
        return cls(
            wcs=wcs,
            geometry=CutoutGeometry(tuple(x_bounds), tuple(y_bounds)),
        )

    @classmethod
    def from_pixel(
        cls,
        wcs: WCS,
        x: float,
        y: float,
        *,
        half: int | None = None,
        half_width: int | None = None,
        half_height: int | None = None,
        left: int | None = None,
        right: int | None = None,
        top: int | None = None,
        bottom: int | None = None,
        half_world: u.Quantity | float | None = None,
    ) -> Self:
        """Build a cutout centered on a pixel coordinate.

        Exactly one size form must be given; see
        :func:`~noobfriend.extraction.cutout._geometry.resolve_offsets` for the
        forms.

        Parameters
        ----------
        wcs : gwcs.WCS
            The target WCS.
        x, y : float
            Center pixel coordinates.
        half, half_width, half_height, left, right, top, bottom, half_world
            Size specification; provide exactly one form.

        Returns
        -------
        Cutout
        """
        offsets = cls._resolve_offsets(
            wcs,
            x,
            y,
            half=half,
            half_width=half_width,
            half_height=half_height,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            half_world=half_world,
        )
        return cls(wcs=wcs, geometry=CutoutGeometry.from_center(x, y, *offsets))

    @classmethod
    def from_world(
        cls,
        wcs: WCS,
        ra: float,
        dec: float,
        *,
        half: int | None = None,
        half_width: int | None = None,
        half_height: int | None = None,
        left: int | None = None,
        right: int | None = None,
        top: int | None = None,
        bottom: int | None = None,
        half_world: u.Quantity | float | None = None,
    ) -> Self:
        """Build a cutout centered on a world coordinate.

        The center ``(ra, dec)`` is mapped to a pixel via ``wcs``; size handling
        is identical to :meth:`from_pixel`.

        Parameters
        ----------
        wcs : gwcs.WCS
            The target WCS.
        ra, dec : float
            Center world coordinates in degrees.
        half, half_width, half_height, left, right, top, bottom, half_world
            Size specification; provide exactly one form.

        Returns
        -------
        Cutout

        Notes
        -----
        For a grism (WFSS) WCS the center maps through the undispersed
        ``detector <-> world`` astrometry, so the cutout is anchored at the
        source's **direct-image** position — not its dispersed spectrum, which
        is offset along the dispersion direction. This is intended: cutout
        geometry is mode-agnostic and spectral extraction is handled elsewhere.
        """
        world_to_detector, _ = world_detector_transforms(wcs)
        x, y = world_to_detector(ra, dec)
        offsets = cls._resolve_offsets(
            wcs,
            x,
            y,
            half=half,
            half_width=half_width,
            half_height=half_height,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            half_world=half_world,
        )
        return cls(wcs=wcs, geometry=CutoutGeometry.from_center(x, y, *offsets))

    @staticmethod
    def _resolve_offsets(
        wcs: WCS,
        x: float,
        y: float,
        *,
        half: int | None,
        half_width: int | None,
        half_height: int | None,
        left: int | None,
        right: int | None,
        top: int | None,
        bottom: int | None,
        half_world: u.Quantity | float | None,
    ) -> tuple[int, int, int, int]:
        """Resolve size kwargs to offsets, computing pixel scale only if needed."""
        scale: tuple[float, float] | None = None
        if half_world is not None:
            _, detector_to_world = world_detector_transforms(wcs)
            scale = pixel_scale_per_deg(
                detector_to_world, round_index(x), round_index(y)
            )
        return resolve_offsets(
            half=half,
            half_width=half_width,
            half_height=half_height,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            half_world=half_world,
            scale=scale,
        )

    @property
    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Integer ``(grid_x, grid_y)`` pixel-index arrays for the region."""
        return self.geometry.meshgrid

    @property
    def meshgrid_world(self) -> tuple[np.ndarray, np.ndarray]:
        """World ``(ra_grid, dec_grid)`` arrays for every pixel in the region."""
        grid_x, grid_y = self.geometry.meshgrid
        _, detector_to_world = world_detector_transforms(self.wcs)
        return detector_to_world(grid_x, grid_y)

    @property
    def footprint(self) -> list[tuple[float, float]]:
        """World ``(ra, dec)`` coordinates of the four cutout corners."""
        _, detector_to_world = world_detector_transforms(self.wcs)
        return [detector_to_world(x, y) for x, y in self.geometry.corners]

    def array(
        self,
        data: np.ndarray,
        wcs: WCS | None = None,
        *,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """Extract the cutout region from ``data`` by an axis-aligned slice.

        No resampling is done: this is a fast integer slice with out-of-range
        padding. For data on a *different* WCS, ``wcs`` shifts the slice window
        by the integer pixel offset between the two frames so the same sky point
        stays centered; for true resampling onto the cutout grid, use
        :meth:`reproject`.

        Parameters
        ----------
        data : numpy.ndarray
            The source array to cut from (a full mosaic or exposure).
        wcs : gwcs.WCS, optional
            The WCS of ``data``. If given and different from :attr:`wcs`, the
            slice window is shifted to compensate for an integer-pixel offset
            between the two frames. If ``None``, ``data`` is assumed to share
            :attr:`wcs`.
        fill_value : float, optional
            Value for pixels outside ``data``, by default :data:`numpy.nan`.

        Returns
        -------
        numpy.ndarray
            The extracted cutout, of shape :attr:`CutoutGeometry.shape`.
        """
        x_bounds, y_bounds = self.geometry.x_bounds, self.geometry.y_bounds
        delta_x = 0
        delta_y = 0
        if wcs is not None and wcs is not self.wcs:
            world_to_detector_data, _ = world_detector_transforms(wcs)
            _, detector_to_world_ref = world_detector_transforms(self.wcs)
            x_ref = (x_bounds[0] + x_bounds[1]) / 2
            y_ref = (y_bounds[0] + y_bounds[1]) / 2
            ra_ref, dec_ref = detector_to_world_ref(x_ref, y_ref)
            x_target, y_target = world_to_detector_data(ra_ref, dec_ref)
            delta_x = round_index(x_target - x_ref)
            delta_y = round_index(y_target - y_ref)

        return slice_with_fill(
            data,
            (x_bounds[0] + delta_x, x_bounds[1] + delta_x),
            (y_bounds[0] + delta_y, y_bounds[1] + delta_y),
            fill_value=fill_value,
        )

    def reproject(
        self,
        data: np.ndarray,
        wcs: WCS,
        *,
        coarse_step: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reproject ``data`` onto the cutout grid, conserving surface brightness.

        Unlike :meth:`array`, this resamples ``data`` (on its own ``wcs``) onto
        this cutout's pixel grid via exact polygon clipping, so it is correct
        when the two WCSs differ in scale, rotation, or distortion. Backed by
        :func:`noobase.image.reproject_exact`.

        Parameters
        ----------
        data : numpy.ndarray
            The source image. Any float dtype is accepted (JWST big-endian
            ``>f4`` included); it is coerced to native-endian float as needed.
            NaNs are treated as masked input pixels.
        wcs : gwcs.WCS
            The WCS of ``data`` (the source frame). Required.
        coarse_step : tuple[int, int], optional
            Evaluate the WCS on a coarse subgrid and bicubic-upsample the
            corners; recommended for JWST gwcs. Each component must divide the
            corresponding axis of :attr:`CutoutGeometry.shape` exactly.

        Returns
        -------
        image : numpy.ndarray
            The reprojected cutout (``float64``); empty pixels are ``NaN``.
        footprint : numpy.ndarray
            Geometric overlap fraction in ``[0, 1]``.
        weight : numpy.ndarray
            Overlap fraction over non-NaN input pixels, ``<= footprint``.

        Notes
        -----
        The cutout grid is this cutout's sub-window of :attr:`wcs`; the target
        pixel-to-world callable is offset by the cutout's lower bounds so output
        pixel ``(0, 0)`` maps to ``(x_start, y_start)`` in the full frame.

        Both mappings use each WCS's ``detector`` <-> ``world`` transforms
        (``get_transform``) rather than the APE-14 ``pixel_to_world_values`` /
        ``world_to_pixel_values``. For grism the APE-14 methods are mode-specific
        (they take extra spectral arguments and resolve to the *dispersed*
        ``grism_detector`` frame), whereas ``detector <-> world`` is the
        undispersed imaging astrometry shared by grism and direct (CLEAR)
        products — so cutouts treat both modes identically.

        There is **no guard on instrument mode**: grism and CLEAR WCSs run
        through the same path. Whether the result is *physically* meaningful
        depends on what ``data``'s pixel values represent, not on the target:

        - A direct image (pixel value = surface brightness) is faithfully
          resampled onto any target grid, including a grism frame's world grid
          (a common WFSS step, valid to first order).
        - Dispersed grism *data* is only meaningful here when the target shares
          the same dispersion (e.g. another exposure of the same field, grism,
          and position angle), where the common dispersion cancels under the
          alignment. Reprojecting dispersed data onto a non-co-dispersed grid is
          geometrically well-defined but is not a surface-brightness map.
        """
        x_start = self.geometry.x_bounds[0]
        y_start = self.geometry.y_bounds[0]
        _, target_detector_to_world = world_detector_transforms(self.wcs)
        source_world_to_detector, _ = world_detector_transforms(wcs)

        def target_pixel_to_world(x: np.ndarray, y: np.ndarray) -> tuple:
            return target_detector_to_world(x + x_start, y + y_start)

        return reproject_onto(
            data,
            target_pixel_to_world=target_pixel_to_world,
            source_world_to_pixel=source_world_to_detector,
            shape=self.geometry.shape,
            coarse_step=coarse_step,
        )
