"""HTML summaries for prepared fit workspaces."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.line import NoobLine
    from noobfriend.inference.spectrum.line.rules import _ParameterRule
    from noobfriend.inference.spectrum.workspace.handles import LineHandle
    from noobfriend.inference.spectrum.workspace.workspace import NoobFitWorkspace


def render_workspace_summary(workspace: NoobFitWorkspace) -> str:
    """Render a read-only HTML summary for a prepared workspace."""
    spectrum = workspace.spectrum
    target_ids = {id(handle.line): handle.id for handle in workspace.handles}
    rows = "\n".join(_line_row(handle, target_ids) for handle in workspace.handles)
    continuum_formula = _continuum_formula(workspace.continuum.order)
    resolving_power = "None" if spectrum.resolving_power is None else _fmt(spectrum.resolving_power)
    z = "None" if spectrum.z is None else _fmt(spectrum.z)
    wavelength_range = f"{_fmt(float(spectrum.obs[0]))} - {_fmt(float(spectrum.obs[-1]))} {spectrum.unit}"

    return f"""<section class="noobfit-summary">
  <style>
    .noobfit-summary {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2933;
      line-height: 1.4;
    }}
    .noobfit-summary h2, .noobfit-summary h3 {{ margin: 0.75rem 0 0.35rem; }}
    .noobfit-summary table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; }}
    .noobfit-summary th, .noobfit-summary td {{
      border: 1px solid #d9e2ec;
      padding: 0.35rem 0.45rem;
      text-align: left;
      vertical-align: top;
    }}
    .noobfit-summary th {{ background: #f0f4f8; }}
    .noobfit-summary code {{ background: #f0f4f8; padding: 0.05rem 0.2rem; border-radius: 3px; }}
    .noobfit-summary .note {{ color: #52606d; }}
  </style>
  <h2>NoobFitWorkspace</h2>
  <p>
    <strong>Lines:</strong> {len(workspace.handles)};
    <strong>ID mode:</strong> {_html(workspace.id_mode)};
    <strong>Valid pixels:</strong> {int(spectrum.valid_mask.sum())}/{spectrum.obs.size}
  </p>
  <p>
    <strong>Wavelength:</strong> {_html(wavelength_range)};
    <strong>z:</strong> {_html(z)};
    <strong>Resolving power:</strong> {_html(resolving_power)}
  </p>
  <h3>Continuum</h3>
  <p><code>{_html(continuum_formula)}</code>, lambda_0 = {_html(_fmt(workspace.continuum.lambda_0))}</p>
  <p class="note">
    Center locks share velocity offset. Wavelength offsets are converted by each line's nominal observed center.
  </p>
  <h3>Lines</h3>
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Line</th>
        <th>Component</th>
        <th>Contribution</th>
        <th>Profile</th>
        <th>Observed center</th>
        <th>Base</th>
        <th>Center rule</th>
        <th>FWHM rule</th>
        <th>Flux rule</th>
      </tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
</section>"""


def _line_row(handle: LineHandle, target_ids: dict[int, str]) -> str:
    line = handle.line
    linename = "<anonymous>" if line.linename is None else line.linename
    rest = "None" if handle.rest_wavelength is None else _fmt(handle.rest_wavelength)
    line_label = f"{_html(linename)}<br><span class=\"note\">rest={_html(rest)}</span>"
    base = "None" if line.base is None else target_ids[id(line.base)]
    return f"""      <tr>
        <td><code>{_html(handle.id)}</code></td>
        <td>{line_label}</td>
        <td>{_html(handle.component)}</td>
        <td>{_html(handle.contribution)}</td>
        <td>{_html(handle.profile)}</td>
        <td>{_html(_fmt(handle.observed_wavelength))}</td>
        <td>{_html(base)}</td>
        <td>{_html(_rule_text(line.center_rule, target_ids, owner=line, kind="center"))}</td>
        <td>{_html(_rule_text(line.fwhm_rule, target_ids, owner=line, kind="fwhm"))}</td>
        <td>{_html(_rule_text(line.flux_rule, target_ids, owner=line, kind="flux"))}</td>
      </tr>"""


def _rule_text(rule: _ParameterRule, target_ids: dict[int, str], *, owner: NoobLine, kind: str) -> str:
    if rule.is_free:
        return "free"
    if rule.is_locked:
        return f"locked to {_target_id(rule, target_ids)}"
    if rule.is_ratio:
        if owner.base is None:
            raise RuntimeError("ratio rule invariant violated: missing owner base.")
        target = target_ids[id(owner.base)]
        if rule.value is not None:
            return f"ratio {_fmt(rule.value)} to {target}"
        raise RuntimeError("ratio rule invariant violated: missing value.")
    if rule.is_fixed:
        if rule.value is None:
            raise RuntimeError("fixed rule invariant violated: missing value.")
        return f"fixed {_rule_value(rule.value, kind=kind, offset_unit=rule.offset_unit)}"
    if rule.is_bounded:
        if rule.bounds is None:
            raise RuntimeError("bounded rule invariant violated: missing bounds.")
        return f"bounded {_rule_value(rule.bounds, kind=kind, offset_unit=rule.offset_unit)}"
    raise RuntimeError("unsupported parameter rule.")


def _rule_value(value: float | tuple[float, float], *, kind: str, offset_unit: str | None) -> str:
    if isinstance(value, tuple):
        text = _bounds(value)
    else:
        text = _fmt(value)
    if kind == "center":
        if offset_unit == "km/s":
            return f"delta_v_kms={text}"
        if offset_unit == "wavelength":
            return f"delta_wavelength={text}"
    if kind == "fwhm":
        return f"{text} km/s"
    return text


def _target_id(rule: _ParameterRule, target_ids: dict[int, str]) -> str:
    if rule.target is None:
        raise RuntimeError("dependent rule invariant violated: missing target.")
    return target_ids[id(rule.target)]


def _continuum_formula(order: int) -> str:
    if order == 0:
        return "c"
    terms = ["c", *(f"k{degree} * (lambda - lambda_0)^{degree}" for degree in range(1, order + 1))]
    terms[1] = "k1 * (lambda - lambda_0)"
    return " + ".join(terms)


def _bounds(bounds: tuple[float, float]) -> str:
    return f"[{_fmt(bounds[0])}, {_fmt(bounds[1])}]"


def _fmt(value: float) -> str:
    return f"{value:.8g}"


def _html(value: object, *, quote: bool = True) -> str:
    return escape(str(value), quote=quote)
