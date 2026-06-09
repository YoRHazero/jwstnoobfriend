"""Interactive Bokeh plots for inspecting images and sky footprints in notebooks.

Two entry points, both returning Bokeh models that render automatically as a
notebook cell's last expression (or via :func:`bokeh.io.show`):

- :func:`imshow` — a zoomable, hoverable view of a 2-D detector frame with a
  percentile stretch and an optional RA/Dec hover read-out.
- :func:`imshow_blink` — several frames overlaid at their own offsets with a
  segmented control (and optional auto-blink) to flip between them, like a
  blink comparator.
- :func:`plot_footprints` — sky footprints drawn on an all-sky Mollweide
  projection, with optional source positions over-plotted.

Render mode defaults to ``"html"`` (a self-contained iframe, robust in VS Code);
call :func:`set_render_mode` with ``"widget"`` for the jupyter_bokeh path.
"""

from noobfriend.core.display.plot._bokeh import set_render_mode
from noobfriend.core.display.plot._footprint import plot_footprints
from noobfriend.core.display.plot._image import imshow
from noobfriend.core.display.plot._blink import imshow_blink

__all__ = ["imshow", "imshow_blink", "plot_footprints", "set_render_mode"]
