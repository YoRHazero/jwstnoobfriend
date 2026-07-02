"""Pixel-space geometry for cutouts: size resolution and resolved bounds.

This module is deliberately WCS-free. It turns a *size specification* (one of
several mutually exclusive forms) into concrete left/right/top/bottom pixel
offsets, and wraps a resolved pair of integer bounds in an immutable
:class:`CutoutGeometry` that can produce its own meshgrid and corner indices.
Anything that needs world coordinates lives in
:mod:`noobfriend.extraction.cutout._core`.

All names here are internal; the only public surface of the subpackage is
:class:`noobfriend.extraction.cutout.Cutout`.
"""

from dataclasses import dataclass

import numpy as np
from astropy import units as u


def round_index(value: float) -> int:
    """Round a pixel coordinate to the nearest integer index.

    Parameters
    ----------
    value : float
        A (possibly fractional) pixel coordinate.

    Returns
    -------
    int
        The nearest integer index.
    """
    return int(np.round(value))


def resolve_offsets(
    *,
    half: int | None = None,
    half_width: int | None = None,
    half_height: int | None = None,
    left: int | None = None,
    right: int | None = None,
    top: int | None = None,
    bottom: int | None = None,
    half_world: u.Quantity | float | None = None,
    scale: tuple[float, float] | None = None,
) -> tuple[int, int, int, int]:
    """Resolve a size specification into ``(left, right, top, bottom)`` offsets.

    Exactly one of the following mutually exclusive size forms must be given:

    - ``left`` + ``right`` + ``top`` + ``bottom`` — explicit per-side offsets.
    - ``half_width`` + ``half_height`` — symmetric per-axis half-extents.
    - ``half`` — a square half-extent in pixels.
    - ``half_world`` — a square half-extent on the sky (requires ``scale``).

    Parameters
    ----------
    half : int, optional
        Square half-extent in pixels.
    half_width, half_height : int, optional
        Symmetric half-extents along x and y, in pixels. Provide both or
        neither.
    left, right, top, bottom : int, optional
        Explicit per-side offsets in pixels. Provide all four or none.
    half_world : astropy.units.Quantity or float, optional
        Square half-extent on the sky. A bare ``float`` is interpreted as
        degrees. Requires ``scale``.
    scale : tuple[float, float], optional
        ``(pixels_per_deg_x, pixels_per_deg_y)``, used only to convert
        ``half_world`` to pixels.

    Returns
    -------
    tuple[int, int, int, int]
        The resolved ``(left, right, top, bottom)`` offsets in pixels.

    Raises
    ------
    ValueError
        If no size form is given, more than one is given, a form is only
        partially specified, or ``half_world`` is given without ``scale``.
    """
    offset_args = (left, right, top, bottom)
    forms_present: list[str] = []
    if any(value is not None for value in offset_args):
        forms_present.append("left/right/top/bottom")
    if half_width is not None or half_height is not None:
        forms_present.append("half_width/half_height")
    if half is not None:
        forms_present.append("half")
    if half_world is not None:
        forms_present.append("half_world")

    if not forms_present:
        raise ValueError(
            "No size specified. Provide one of: left/right/top/bottom, "
            "half_width/half_height, half, or half_world."
        )
    if len(forms_present) > 1:
        raise ValueError(
            f"Size is over-specified; provide exactly one form, got {forms_present}."
        )

    if forms_present[0] == "left/right/top/bottom":
        if any(value is None for value in offset_args):
            raise ValueError(
                "left, right, top, and bottom must all be provided together."
            )
        return int(left), int(right), int(top), int(bottom)

    if forms_present[0] == "half_width/half_height":
        if half_width is None or half_height is None:
            raise ValueError("half_width and half_height must be provided together.")
        return int(half_width), int(half_width), int(half_height), int(half_height)

    if forms_present[0] == "half":
        value = int(half)
        return value, value, value, value

    # half_world
    if scale is None:
        raise ValueError("half_world requires a pixel scale to convert to pixels.")
    half_deg = (
        half_world.to(u.deg).value
        if isinstance(half_world, u.Quantity)
        else float(half_world)
    )
    half_w = int(np.ceil(half_deg * scale[0]))
    half_h = int(np.ceil(half_deg * scale[1]))
    return half_w, half_w, half_h, half_h


@dataclass(frozen=True)
class CutoutGeometry:
    """An immutable cutout region as half-open integer pixel bounds.

    Both bounds follow Python slice semantics: ``x_bounds = (x_start, x_end)``
    covers columns ``x_start .. x_end - 1`` (likewise for ``y_bounds`` and
    rows).

    Attributes
    ----------
    x_bounds : tuple[int, int]
        ``(x_start, x_end)`` column bounds, half-open.
    y_bounds : tuple[int, int]
        ``(y_start, y_end)`` row bounds, half-open.
    """

    x_bounds: tuple[int, int]
    y_bounds: tuple[int, int]

    @classmethod
    def from_center(
        cls, x: float, y: float, left: int, right: int, top: int, bottom: int
    ) -> "CutoutGeometry":
        """Build bounds from a (possibly fractional) center and pixel offsets.

        The center is rounded to the nearest integer pixel, then the offsets are
        applied so that ``half == left == right`` yields ``2 * half + 1`` columns
        centered on that pixel.

        Parameters
        ----------
        x, y : float
            Center pixel coordinates; rounded to the nearest integer.
        left, right, top, bottom : int
            Per-side offsets in pixels.

        Returns
        -------
        CutoutGeometry
            The resolved bounds.
        """
        center_x = round_index(x)
        center_y = round_index(y)
        return cls(
            x_bounds=(center_x - left, center_x + right + 1),
            y_bounds=(center_y - bottom, center_y + top + 1),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_rows, n_cols)`` shape of the cutout, matching numpy convention."""
        return (
            self.y_bounds[1] - self.y_bounds[0],
            self.x_bounds[1] - self.x_bounds[0],
        )

    def padded_to_multiple(self, row_step: int, col_step: int) -> "CutoutGeometry":
        """Grow the bounds minimally so each axis length is a multiple of its step.

        Coarse-grid reprojection (``coarse_step``) evaluates the WCS on a
        sub-sampled corner field, which requires the step to divide the output
        shape exactly. This pads the region outward to the next such multiple,
        splitting the extra pixels across the two sides of each axis (the low
        side takes the floor half) so the centre barely moves. The padding on an
        axis is always ``< step``; a step ``<= 1`` leaves that axis unchanged.

        Parameters
        ----------
        row_step, col_step : int
            The row (y) and column (x) coarse steps the shape must divide by --
            i.e. ``(coarse_step[0], coarse_step[1])``.

        Returns
        -------
        CutoutGeometry
            A new geometry whose ``shape`` is a multiple of ``(row_step,
            col_step)`` on each axis.
        """

        def grow(lo: int, hi: int, step: int) -> tuple[int, int]:
            if step <= 1:
                return lo, hi
            pad = (-(hi - lo)) % step
            return lo - pad // 2, hi + (pad - pad // 2)

        return CutoutGeometry(
            x_bounds=grow(*self.x_bounds, col_step),
            y_bounds=grow(*self.y_bounds, row_step),
        )

    @property
    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Integer ``(grid_x, grid_y)`` pixel-index arrays for the region.

        Returns
        -------
        grid_x, grid_y : numpy.ndarray
            Column- and row-index arrays of shape :attr:`shape`.
        """
        (x_start, x_end), (y_start, y_end) = self.x_bounds, self.y_bounds
        grid_y, grid_x = np.mgrid[y_start:y_end, x_start:x_end]
        return grid_x, grid_y

    @property
    def corners(self) -> list[tuple[int, int]]:
        """The four ``(x, y)`` corner indices, counter-clockwise from bottom-left.

        Uses the exclusive upper bounds, so the corners enclose the full pixel
        footprint rather than the centres of the edge pixels.
        """
        (x_start, x_end), (y_start, y_end) = self.x_bounds, self.y_bounds
        return [
            (x_start, y_start),
            (x_end, y_start),
            (x_end, y_end),
            (x_start, y_end),
        ]
