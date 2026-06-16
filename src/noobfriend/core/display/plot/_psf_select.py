"""Interactive Bokeh panel for selecting PSF stars off the size-flux locus.

Three linked panels -- a ``flux`` vs ``fwhm`` scatter coloured by the cross-band
``max_ell``, a controls column, and (optionally) a stamp-preview panel that shows
the hovered source's thumbnails. The user sets the locus with range sliders and
overrides individual sources by tapping; the panel reports how many sources are
**selected**, plus how many are *extra* **included** (manually kept though
outside the slider ranges) and *extra* **excluded** (manually dropped though
inside the ranges), and offers a copy button for each list of indices.

No Python round-trip: everything runs client-side, so the read-out values and
the copied index lists are typed back into
:meth:`~noobfriend.extraction.psf.SourceExtractor.select` (``fwhm`` / ``flux`` /
``max_ellipticity`` ranges, ``include`` / ``exclude`` lists). Pass
``show_thumbs=False`` for a large catalogue to skip embedding thumbnails.
"""

import base64
import io

import numpy as np
from bokeh.events import ButtonClick
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    LinearColorMapper,
    RangeSlider,
    Slider,
    Span,
)
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

from noobfriend.core.display.plot._bokeh import display

# Client-side recompute of selection state from the current slider values, shared
# by the slider and tap callbacks (args: src, fw, fl, el, count, fw_lo..fl_hi).
_RECOMPUTE = """
const d = src.data;
const wlo = fw.value[0], whi = fw.value[1];
const flo = Math.pow(10, fl.value[0]), fhi = Math.pow(10, fl.value[1]);
const emax = el.value;
let nsel = 0, ninc = 0, nexc = 0;
for (let i = 0; i < d['idx'].length; i++) {
  const inr = d['fwhm'][i] >= wlo && d['fwhm'][i] <= whi
           && d['flux'][i] >= flo && d['flux'][i] <= fhi
           && d['max_ell'][i] <= emax;
  const ov = d['override'][i];
  const sel = ov === 1 ? true : (ov === -1 ? false : inr);
  d['alpha'][i] = sel ? 0.95 : 0.06;
  d['ring'][i] = ov === 1 ? '#1a9850' : (ov === -1 ? '#d73027' : 'rgba(0,0,0,0)');
  if (sel) nsel++;
  if (ov === 1 && !inr) ninc++;
  if (ov === -1 && inr) nexc++;
}
fw_lo.location = wlo; fw_hi.location = whi; fl_lo.location = flo; fl_hi.location = fhi;
src.change.emit();
count.text = '<b>selected: ' + nsel + '</b> / ' + d['idx'].length
  + '<br>included (extra): <b>' + ninc + '</b>'
  + '<br>excluded (extra): <b>' + nexc + '</b>';
"""

_TAP = """
const picked = src.selected.indices;
if (picked.length === 0) { return; }
const dd = src.data;
const twlo = fw.value[0], twhi = fw.value[1];
const tflo = Math.pow(10, fl.value[0]), tfhi = Math.pow(10, fl.value[1]);
const temax = el.value;
for (const i of picked) {
  if (dd['override'][i] !== 0) {
    dd['override'][i] = 0;
  } else {
    const inr = dd['fwhm'][i] >= twlo && dd['fwhm'][i] <= twhi
             && dd['flux'][i] >= tflo && dd['flux'][i] <= tfhi
             && dd['max_ell'][i] <= temax;
    dd['override'][i] = inr ? -1 : 1;  // in range -> exclude, out of range -> include
  }
}
src.selected.indices = [];
"""


def _thumb_png(array: np.ndarray) -> str:
    """Render one cutout to a crisp asinh-stretched base64 PNG data URI."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.asarray(array, dtype=float)
    a = a - np.nanmedian(a)
    vmax = np.nanpercentile(a, 99.5)
    vmax = vmax if vmax > 0 else 1.0
    fig = plt.figure(figsize=(1.0, 1.0), dpi=64)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.imshow(
        np.arcsinh(a / vmax * 12.0),
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _thumb_row_html(arrays: list[np.ndarray], labels: list[str]) -> str:
    """Build the hover-preview HTML: a wrapping row of labelled thumbnails."""
    cells = [
        f'<div style="text-align:center;margin:4px">'
        f'<img src="{_thumb_png(arr)}" style="width:150px;height:150px;'
        f'image-rendering:pixelated;border:1px solid #ccc">'
        f'<div style="font-size:12px;color:#333">{label}</div></div>'
        for arr, label in zip(arrays, labels)
    ]
    return (
        '<div style="display:flex;flex-wrap:wrap;justify-content:center">'
        + "".join(cells)
        + "</div>"
    )


def _copy_code(keep: str, label: str) -> str:
    """Return the clipboard CustomJS for a copy button (keeps where ``keep``)."""
    return f"""
    const d = src.data, out = [];
    const wlo = fw.value[0], whi = fw.value[1];
    const flo = Math.pow(10, fl.value[0]), fhi = Math.pow(10, fl.value[1]);
    const emax = el.value;
    for (let i = 0; i < d['idx'].length; i++) {{
      const inr = d['fwhm'][i] >= wlo && d['fwhm'][i] <= whi
               && d['flux'][i] >= flo && d['flux'][i] <= fhi
               && d['max_ell'][i] <= emax;
      if ({keep}) out.push(d['idx'][i]);
    }}
    const text = '[' + out.join(', ') + ']';
    const ok = () => {{ status.text = 'copied {label}: ' + out.length; }};
    const fallback = () => {{
      const ta = document.createElement('textarea');
      ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.focus(); ta.select();
      let done = false; try {{ done = document.execCommand('copy'); }} catch (e) {{}}
      document.body.removeChild(ta);
      status.text = done ? ('copied {label}: ' + out.length) : 'copy failed - select & Ctrl-C';
    }};
    if (navigator.clipboard && navigator.clipboard.writeText) {{
      navigator.clipboard.writeText(text).then(ok, fallback);
    }} else {{ fallback(); }}
    """


def psf_select_panel(
    idx: np.ndarray,
    flux: np.ndarray,
    fwhm: np.ndarray,
    ell: np.ndarray,
    max_ell: np.ndarray,
    snr: np.ndarray,
    *,
    thumbs: list[list[np.ndarray]] | None = None,
    thumb_labels: list[list[str]] | None = None,
    show_thumbs: bool = True,
    height: int = 820,
):
    """Return the interactive PSF-star selection panel (per physical source).

    Parameters
    ----------
    idx : numpy.ndarray
        Physical-source index of each point (what ``select(include=/exclude=)``
        consumes).
    flux, fwhm, ell, snr : numpy.ndarray
        Per-source values in the band being plotted (medians).
    max_ell : numpy.ndarray
        Per-source maximum ellipticity across all bands -- the cross-band
        point-likeness, used as the colour and the ellipticity cut.
    thumbs : list of list of numpy.ndarray, optional
        Per source, the cutout arrays to preview (e.g. one per band); shown only
        when ``show_thumbs``.
    thumb_labels : list of list of str, optional
        Per source, a label for each thumbnail (e.g. the band name).
    show_thumbs : bool, default True
        Embed thumbnails (a hover-switching preview panel). Pass ``False`` for a
        large catalogue to avoid rendering thousands of images.
    height : int, default 700
        Iframe height for the returned display handle.

    Returns
    -------
    A notebook-displayable Bokeh handle (also valid as a cell's last expression).
    """
    idx = np.asarray(idx)
    flux = np.asarray(flux, dtype=float)
    fwhm = np.asarray(fwhm, dtype=float)
    ell = np.asarray(ell, dtype=float)
    max_ell = np.asarray(max_ell, dtype=float)
    snr = np.asarray(snr, dtype=float)
    n = idx.size

    fwhm_lo, fwhm_hi = (float(fwhm.min()), float(fwhm.max())) if n else (0.0, 1.0)
    flux_lo, flux_hi = (float(flux.min()), float(flux.max())) if n else (1.0, 10.0)
    ell_init = float(min(0.15, max_ell.max())) if n else 0.15
    size = np.clip(3.0 + np.log10(np.clip(snr, 1, None)) * 4.0, 3.0, 18.0)
    selected0 = (max_ell <= ell_init) if n else np.zeros(0, dtype=bool)

    data = dict(
        idx=idx,
        flux=flux,
        fwhm=fwhm,
        ell=ell,
        max_ell=max_ell,
        snr=snr,
        size=size,
        override=np.zeros(n, dtype=int),
        alpha=np.where(selected0, 0.95, 0.06),
        ring=["rgba(0,0,0,0)"] * n,
    )
    if show_thumbs and thumbs is not None:
        labels = thumb_labels or [[""] * len(t) for t in thumbs]
        data["thumbs_html"] = [_thumb_row_html(thumbs[i], labels[i]) for i in range(n)]
    src = ColumnDataSource(data)

    cmap = LinearColorMapper(palette=Viridis256, low=0.0, high=0.3)
    plot = figure(
        width=560,
        height=470,
        x_axis_type="log",
        x_axis_label="flux",
        y_axis_label="fwhm [pix]",
        title="size-flux  (colour = max ellipticity across bands, size = SNR)",
        tools="pan,wheel_zoom,box_zoom,tap,reset,save",
    )
    glyph = plot.scatter(
        "flux",
        "fwhm",
        source=src,
        size="size",
        line_color="ring",
        line_width=2,
        fill_color={"field": "max_ell", "transform": cmap},
        fill_alpha="alpha",
    )
    plot.add_layout(ColorBar(color_mapper=cmap, title="max ellipticity"), "right")
    plot.add_tools(
        HoverTool(
            renderers=[glyph],
            tooltips=[
                ("source", "@idx"),
                ("snr", "@snr{0.0}"),
                ("fwhm", "@fwhm{0.00}"),
                ("ell", "@ell{0.000}"),
                ("max_ell", "@max_ell{0.000}"),
            ],
        )
    )

    fw_lo = Span(
        dimension="width", location=fwhm_lo, line_dash="dashed", line_color="#888"
    )
    fw_hi = Span(
        dimension="width", location=fwhm_hi, line_dash="dashed", line_color="#888"
    )
    fl_lo = Span(
        dimension="height", location=flux_lo, line_dash="dashed", line_color="#888"
    )
    fl_hi = Span(
        dimension="height", location=flux_hi, line_dash="dashed", line_color="#888"
    )
    for span in (fw_lo, fw_hi, fl_lo, fl_hi):
        plot.add_layout(span)

    fw = RangeSlider(
        start=fwhm_lo,
        end=fwhm_hi,
        value=(fwhm_lo, fwhm_hi),
        step=0.05,
        title="fwhm range",
    )
    fl = RangeSlider(
        start=float(np.log10(flux_lo)),
        end=float(np.log10(flux_hi)),
        value=(float(np.log10(flux_lo)), float(np.log10(flux_hi))),
        step=0.05,
        title="log10 flux range",
    )
    el = Slider(start=0.0, end=0.3, value=ell_init, step=0.005, title="max ellipticity")
    count = Div(
        text=f"<b>selected: {int(selected0.sum())}</b> / {n}"
        "<br>included (extra): <b>0</b><br>excluded (extra): <b>0</b>",
        styles={"font-size": "14px"},
    )
    status = Div(styles={"font-size": "12px", "color": "#2a7"})

    span_args = dict(fw_lo=fw_lo, fw_hi=fw_hi, fl_lo=fl_lo, fl_hi=fl_hi)
    recolor = CustomJS(
        args=dict(src=src, fw=fw, fl=fl, el=el, count=count, **span_args),
        code=_RECOMPUTE,
    )
    for control in (fw, fl, el):
        control.js_on_change("value", recolor)
    src.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(src=src, fw=fw, fl=fl, el=el, count=count, **span_args),
            code=_TAP + _RECOMPUTE,
        ),
    )

    copy_inc = Button(label="Copy included indices", button_type="success", width=220)
    copy_inc.js_on_event(
        ButtonClick,
        CustomJS(
            args=dict(src=src, fw=fw, fl=fl, el=el, status=status),
            code=_copy_code("d['override'][i] === 1 && !inr", "included"),
        ),
    )
    copy_exc = Button(label="Copy excluded indices", button_type="danger", width=220)
    copy_exc.js_on_event(
        ButtonClick,
        CustomJS(
            args=dict(src=src, fw=fw, fl=fl, el=el, status=status),
            code=_copy_code("d['override'][i] === -1 && inr", "excluded"),
        ),
    )

    header = Div(
        text=(
            "<h3>PSF star selection</h3>"
            "<p style='font-size:12px'>Drag the range sliders to set the locus; "
            "tap a point to override (in-range tap excludes, out-of-range tap "
            "includes). Copy the included / excluded index lists and read the "
            "slider ranges into <code>select(...)</code>.</p>"
        ),
        width=1100,
    )
    controls = column(fw, fl, el, count, copy_inc, copy_exc, status)

    if show_thumbs and thumbs is not None:
        preview = Div(
            width=860,
            height=250,
            text="<div style='color:#888;text-align:center;padding-top:90px'>"
            "hover a point to preview its thumbnails</div>",
        )
        plot.add_tools(
            HoverTool(
                renderers=[glyph],
                tooltips=None,
                callback=CustomJS(
                    args=dict(src=src, preview=preview),
                    code="""
                    const idxs = cb_data.index.indices;
                    if (idxs.length === 0) return;
                    preview.text = src.data['thumbs_html'][idxs[0]];
                    """,
                ),
            )
        )
        # thumbnails go in a wide strip below the scatter + controls, so a
        # horizontal row of (>=3) thumbs has room without crowding the plot.
        layout = column(header, row(plot, controls), preview)
    else:
        layout = column(header, row(plot, controls))

    return display(layout, height=height)
