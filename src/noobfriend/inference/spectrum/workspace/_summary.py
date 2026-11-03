"""Plain-text summary of a workspace's fit plan."""

from __future__ import annotations

from typing import TYPE_CHECKING

from noobfriend.inference.spectrum._text import format_rows

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace._workspace import NoobFitWorkspace


def workspace_summary(workspace: NoobFitWorkspace) -> str:
    """Render the full fit plan as text.

    Sections: data, frames (resolving power, noise, fitted pixels),
    components, the parameter table with every prior or fixed value, derived
    line quantities, and joint priors with their marginalized members.
    """
    spectrum = workspace.spectrum
    scales = workspace.scales
    lines = [
        "NoobFitWorkspace",
        f"  frames: {len(spectrum.frames)}   unit: {spectrum.unit}   z: {spectrum.z}",
        f"  window: {'all usable pixels' if workspace.window is None else workspace.window}",
        f"  data scales (for default priors): flux S={scales.flux:.4g}, "
        f"line amplitude={scales.amplitude:.4g}, half span W={scales.wavelength:.4g}",
        "",
        "Frames",
    ]
    frame_rows = []
    for frame_id, frame in spectrum.frames.items():
        frame_rows.append(
            [
                frame_id,
                str(frame.obs.size),
                str(int(workspace.fitted[frame_id].sum())),
                _resolving_power_text(frame.resolving_power),
                str(frame.noise),
            ]
        )
    lines += format_rows(["frame", "pixels", "fitted", "R", "noise"], frame_rows)

    lines += ["", "Components"]
    component_rows = []
    for handle in workspace.lines:
        lsf = sorted(
            {f"{v:.4g}" for v in handle.lsf_fwhm_kms.values() if v is not None}
        )
        component_rows.append(
            [
                handle.name,
                f"{handle.profile} {handle.component} {handle.contribution}",
                f"{handle.nominal:.8g}",
                handle.base or "-",
                ",".join(handle.kernels) or "-",
                "/".join(lsf) or "none",
            ]
        )
    continuum = workspace.continuum
    shape = f"order {continuum.order}" if continuum.kind == "polynomial" else ""
    component_rows.append(
        [
            "continuum",
            f"{continuum.kind} {shape}".strip(),
            f"lambda_0={continuum.lambda_0:.8g}",
            "-",
            f"sharing={continuum.sharing}",
            "-",
        ]
    )
    lines += format_rows(
        ["name", "shape", f"nominal ({spectrum.unit})", "base", "kernels", "LSF km/s"],
        component_rows,
    )

    lines += ["", "Parameters"]
    lines += format_rows(
        ["name", "unit", "rule"],
        [
            [row.name, row.unit or "-", row.rule()]
            for row in workspace.parameters.values()
        ],
    )

    derived = [q for q in workspace.parameters.quantities.values() if not q.is_direct]
    if derived:
        lines += ["", "Derived line quantities"]
        lines += [f"  {q.name} = {q.expression.text()}" for q in derived]

    if workspace.joint_priors:
        lines += ["", "Joint priors (multivariate normal in unconstrained coordinates)"]
        for index, block in enumerate(workspace.joint_priors, start=1):
            pairs = ", ".join(
                name if name == member else f"{name} <- {member}"
                for name, member in zip(block.parameters, block.members, strict=True)
            )
            lines.append(f"  [{index}] {pairs}")
            if block.marginalized:
                lines.append(f"      marginalized out: {', '.join(block.marginalized)}")
    return "\n".join(lines)


def _resolving_power_text(resolving_power: object) -> str:
    if resolving_power is None:
        return "none"
    if isinstance(resolving_power, tuple):
        values = resolving_power[1]
        return f"table {float(values.min()):.4g}-{float(values.max()):.4g}"
    return f"{resolving_power:.6g}"
