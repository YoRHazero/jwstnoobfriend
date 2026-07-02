"""Interactive Bokeh plots for inspecting images and sky footprints in notebooks.

Two entry points, both returning Bokeh models that render automatically as a
notebook cell's last expression (or via :func:`bokeh.io.show`):

- :func:`imshow` — a zoomable, hoverable view of a 2-D detector frame with a
  percentile stretch, an optional RA/Dec hover read-out, and optional
  :class:`ImageOverlay` instances such as :class:`CatalogOverlay`.
- :func:`imshow_blink` — several frames overlaid at their own offsets with a
  segmented control (and optional auto-blink) to flip between them, like a
  blink comparator.
- :func:`plot_footprints` — sky footprints drawn on an all-sky Mollweide
  projection, with optional source positions over-plotted.

Render mode defaults to ``"html"`` (a self-contained iframe, robust in VS Code);
call :func:`set_render_mode` with ``"widget"`` for the jupyter_bokeh path.

Spectra are the one exception to the Bokeh backend, rendered with static
matplotlib instead (no interactivity needed, trivial ``savefig`` export):

- :func:`plot_spectrum1d` — one or more 1-D spectra with optional model overlays
  described by :class:`ModelSpec`.
- :func:`plot_spectrum2d` — a 2-D spectrum image stacked over its 1-D collapse
  on a shared wavelength axis.
- :func:`draw_spectrum` — the lower-level engine behind both, drawing data and
  model overlays onto a caller-supplied Axes for custom multi-panel figures.
"""

from noobfriend.core.display.plot._bokeh import set_render_mode
from noobfriend.core.display.plot._footprint import plot_footprints
from noobfriend.core.display.plot._image import imshow
from noobfriend.core.display.plot._blink import imshow_blink
from noobfriend.core.display.plot._overlay import (
    CatalogOverlay,
    ImageFrame,
    ImageOverlay,
)
from noobfriend.core.display.plot._psf_select import psf_select_panel
from noobfriend.core.display.plot._spectrum import ModelSpec, draw_spectrum
from noobfriend.core.display.plot._spectrum1d import plot_spectrum1d
from noobfriend.core.display.plot._spectrum2d import plot_spectrum2d

__all__ = [
    "ModelSpec",
    "CatalogOverlay",
    "ImageFrame",
    "ImageOverlay",
    "draw_spectrum",
    "imshow",
    "imshow_blink",
    "plot_footprints",
    "plot_spectrum1d",
    "plot_spectrum2d",
    "psf_select_panel",
    "set_render_mode",
]
