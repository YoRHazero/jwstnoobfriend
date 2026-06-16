"""Shared Bokeh plumbing for the plot subpackage: notebook setup and palettes.

All names here are internal. They centralise the two pieces of Bokeh state that
both :mod:`~noobfriend.core.display.plot._image` and
:mod:`~noobfriend.core.display.plot._footprint` need: a once-only notebook
handshake that loads BokehJS, and palette resolution.
"""

from collections.abc import Sequence
from typing import Any

from bokeh import palettes
from bokeh.io import output_notebook
from bokeh.models import LayoutDOM
from bokeh.palettes import Category10, Category20, viridis

_NOTEBOOK_READY: bool = False
_RENDER_MODE: str = "html"


def set_render_mode(mode: str) -> None:
    """Select how :func:`imshow` / :func:`plot_footprints` return plots for display.

    Parameters
    ----------
    mode : {"widget", "html"}
        ``"html"`` (default) returns a self-contained BokehJS document inside a
        sandboxed ``<iframe>``, which bypasses the Jupyter widget system and
        renders wherever notebook HTML output runs scripts (VS Code included).
        ``"widget"`` instead wraps each figure in a :mod:`jupyter_bokeh`
        ipywidget -- nicer (auto-sizing, live Python links) but renders blank in
        some VS Code setups where third-party widget JS is not fetched.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"widget"`` or ``"html"``.
    """
    global _RENDER_MODE
    if mode not in ("widget", "html"):
        raise ValueError(f"mode must be 'widget' or 'html', got {mode!r}")
    _RENDER_MODE = mode


def ensure_notebook() -> None:
    """Load BokehJS into the active notebook exactly once.

    Idempotent: the first call performs the :func:`bokeh.io.output_notebook`
    handshake (which registers the IPython display hook so returned figures
    render automatically); later calls are no-ops. Safe to call from a plain
    script, where it simply primes Bokeh's notebook output mode.
    """
    global _NOTEBOOK_READY
    if _NOTEBOOK_READY:
        return
    output_notebook(hide_banner=True)
    _NOTEBOOK_READY = True


def display(model: LayoutDOM, height: int | None = None) -> Any:
    """Return a notebook-displayable handle for a Bokeh model.

    Prefers the :mod:`jupyter_bokeh` ipywidget path (a ``BokehModel``), which
    renders reliably in VS Code, JupyterLab, and classic notebooks because they
    all support Jupyter widgets. When :mod:`jupyter_bokeh` is not installed,
    falls back to the bare model plus a one-time
    :func:`bokeh.io.output_notebook` handshake -- this renders in
    JupyterLab/classic but commonly fails in VS Code, whose notebook renderer
    does not execute Bokeh's injected ``<script>`` output.

    Parameters
    ----------
    model : bokeh.models.LayoutDOM
        The figure or layout to display.
    height : int, optional
        Iframe height in pixels for ``"html"`` mode. Ignored by the widget
        path (widgets size themselves). Defaults to 760.

    Returns
    -------
    Any
        With ``set_render_mode("html")``, an :class:`IPython.display.HTML`
        iframe. Otherwise a ``jupyter_bokeh.BokehModel`` widget if available,
        else the bare ``model`` (plus an :func:`output_notebook` handshake).
    """
    if _RENDER_MODE == "html":
        return _html_iframe(model, height=height or 760)
    try:
        from jupyter_bokeh import BokehModel
    except ImportError:
        ensure_notebook()
        return model
    return BokehModel(model)


def _html_iframe(model: LayoutDOM, height: int = 760) -> Any:
    """Embed ``model`` as a self-contained BokehJS document in a sandboxed iframe.

    Renders a full standalone HTML document with BokehJS embedded **inline** into
    an ``<iframe srcdoc=...>``, giving the plot its own clean document context.
    Inline (not CDN) resources are used deliberately: VS Code's sandboxed
    ``srcdoc`` iframe fails to load multiple external CDN bundles (a figure with
    widgets pulls in both ``bokeh`` and ``bokeh-widgets``), which left
    widget-bearing plots blank. Inlining has no external scripts, so it renders
    regardless and works offline. This also sidesteps Bokeh's classic
    ``output_notebook`` script injection and the Jupyter widget system.

    Parameters
    ----------
    model : bokeh.models.LayoutDOM
        The figure or layout to embed.
    height : int, default 760
        Iframe height in pixels (figures are ~620 px tall plus toolbar/slider).

    Returns
    -------
    IPython.display.HTML
        The iframe wrapper, displayable as a notebook cell's last expression.
    """
    import warnings
    from html import escape

    from bokeh.embed import file_html
    from bokeh.resources import INLINE
    from IPython.display import HTML

    srcdoc = escape(file_html(model, INLINE), quote=True)
    iframe = (
        f'<iframe srcdoc="{srcdoc}" '
        f'style="width:100%; height:{height}px; border:none;" '
        f'sandbox="allow-scripts allow-same-origin"></iframe>'
    )
    with warnings.catch_warnings():
        # IPython suggests IFrame, but that takes a src URL, not srcdoc.
        warnings.filterwarnings(
            "ignore", message="Consider using IPython.display.IFrame"
        )
        return HTML(iframe)


def resolve_palette(cmap: str | Sequence[str]) -> Sequence[str]:
    """Resolve a colormap name or explicit color list into a Bokeh palette.

    Parameters
    ----------
    cmap : str or sequence of str
        Either the name of a palette exported by :mod:`bokeh.palettes`
        (e.g. ``"Viridis256"``, ``"Greys256"``) or an explicit list of CSS
        color strings to use verbatim.

    Returns
    -------
    sequence of str
        The resolved palette.

    Raises
    ------
    ValueError
        If ``cmap`` is a string that names no known Bokeh palette.
    """
    if not isinstance(cmap, str):
        return list(cmap)
    pal = getattr(palettes, cmap, None)
    if pal is None:
        raise ValueError(f"unknown Bokeh palette {cmap!r}")
    return pal


def categorical_palette(n: int) -> list[str]:
    """Return ``n`` visually distinct colors for labelling categorical items.

    Uses Category10 / Category20 for small counts and falls back to viridis
    beyond 20. When more colors are requested than Bokeh's 256-colour viridis
    function supports, the 256-colour ramp is cycled so callers still receive a
    length-``n`` list instead of an exception.

    Parameters
    ----------
    n : int
        Number of colors requested.

    Returns
    -------
    list of str
        ``n`` color strings (empty if ``n <= 0``).
    """
    if n <= 0:
        return []
    if n <= 10:
        return list(Category10[10])[:n]
    if n <= 20:
        return list(Category20[20])[:n]
    if n <= 256:
        return list(viridis(n))
    base = list(viridis(256))
    return [base[i % len(base)] for i in range(n)]
