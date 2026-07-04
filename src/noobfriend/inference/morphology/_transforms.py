"""Deterministic maps from standard-normal coordinates to physical parameters.

The sampler only ever sees ``z ~ Normal(0, I)``; this module gives each
coordinate its physical meaning. All charts are singularity-free:

- centres are affine maps of ``z``;
- the ellipticity vector ``(e1, e2)`` lives on an open disk of radius
  ``e_max`` via the diffeomorphism ``u -> u / sqrt(1 + |u|^2)`` (no angle
  coordinate, so ``q -> 1`` is a regular point);
- the host offset lives on the open unit disk in normalized
  elliptical-radius units, which enforces hard containment of the primary
  point source inside the one-``r_eff`` host ellipse without a polar chart.

Everything here is JAX-traceable and dependency-light: ``jax.numpy`` is
injected by the caller so this module imports cleanly without JAX.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from noobfriend.inference.morphology.models import ModelSpec
from noobfriend.inference.morphology.priors import Priors

__all__ = ["Transform", "build_transform"]


@dataclass(frozen=True)
class Transform:
    """Bundle of the latent dimension and the z-to-physical map.

    Parameters
    ----------
    dim : int
        Number of latent standard-normal coordinates.
    names : tuple of str
        Names of the physical (and derived) parameters produced by
        ``physical``, in a stable order.
    physical : callable
        ``physical(z, np_like)`` mapping a length-``dim`` vector to a dict of
        named scalars. ``np_like`` is ``jax.numpy`` inside jitted code or
        ``numpy`` for plain evaluation.
    """

    dim: int
    names: tuple[str, ...]
    physical: Callable[[Any, Any], dict[str, Any]]


def saturating(z: Any, cap: float, xp: Any) -> Any:
    """Map the line onto ``(-cap, cap)``, identity-like near zero.

    ``cap * tanh(z / cap)`` keeps the prior interpretation of ``z`` intact
    within roughly ``|z| < cap/2`` while giving scale-like parameters compact
    range. Without the cap, a component unsupported by the data (e.g. a host
    fit to point-only data) has an unbounded flat escape direction — the
    marginal likelihood keeps improving as the component's basis image drains
    off the stamp — and NUTS pays maximal tree depths to traverse it. With
    the cap, the latent tail reverts to the standard-normal prior, which
    samples trivially.
    """
    return cap * xp.tanh(z / cap)


def disk_map(u_x: Any, u_y: Any, xp: Any) -> tuple[Any, Any]:
    """Map the plane diffeomorphically onto the open unit disk.

    Parameters
    ----------
    u_x, u_y : array_like
        Unconstrained coordinates.
    xp : module
        ``numpy`` or ``jax.numpy``.

    Returns
    -------
    tuple
        Disk coordinates ``(d_x, d_y)`` with ``d_x**2 + d_y**2 < 1``.
    """
    norm = xp.sqrt(1.0 + u_x * u_x + u_y * u_y)
    return u_x / norm, u_y / norm


def build_transform(spec: ModelSpec, priors: Priors) -> Transform:
    """Build the latent-to-physical transform for one model.

    Latent layout (in order): two coordinates per point-source centre, then
    for the host: ``log r_eff``, ``log n``, two ellipticity coordinates, and
    two offset coordinates (or two centre coordinates for host-only models).

    Parameters
    ----------
    spec : ModelSpec
        The model whose parameters are being mapped.
    priors : Priors
        Prior locations/scales consumed by the transforms.

    Returns
    -------
    Transform
        The latent dimension, physical parameter names, and the map itself.
    """
    dim = 2 * spec.n_points + (6 if spec.has_host else 0)
    names: list[str] = []
    for k in range(spec.n_points):
        names += [f"point{k}_x", f"point{k}_y"]
    if spec.has_host:
        names += [
            "host_r_eff",
            "host_n",
            "host_e1",
            "host_e2",
            "host_q",
            "host_theta_deg",
            "host_x",
            "host_y",
        ]
        if spec.n_points >= 1:
            names.append("host_offset_rho")

    point = priors.point
    host = priors.host

    def physical(z: Any, xp: Any) -> dict[str, Any]:
        """Map latent coordinates to named physical parameters."""
        out: dict[str, Any] = {}
        idx = 0
        for k in range(spec.n_points):
            center = point.center if k == 0 else point.extra_center
            out[f"point{k}_x"] = center.x_loc + center.scale * z[idx]
            out[f"point{k}_y"] = center.y_loc + center.scale * z[idx + 1]
            idx += 2
        if spec.has_host:
            r_eff = xp.exp(
                host.log_r_eff_loc + host.log_r_eff_scale * saturating(z[idx], 2.5, xp)
            )
            n = xp.exp(
                host.log_n_loc + host.log_n_scale * saturating(z[idx + 1], 2.5, xp)
            )
            e1, e2 = disk_map(host.e_scale * z[idx + 2], host.e_scale * z[idx + 3], xp)
            e1 = host.e_max * e1
            e2 = host.e_max * e2
            e_mag = xp.sqrt(e1 * e1 + e2 * e2)
            q = (1.0 - e_mag) / (1.0 + e_mag)
            theta = 0.5 * xp.arctan2(e2, e1)
            out["host_r_eff"] = r_eff
            out["host_n"] = n
            out["host_e1"] = e1
            out["host_e2"] = e2
            out["host_q"] = q
            out["host_theta_deg"] = xp.rad2deg(theta)
            if spec.n_points >= 1:
                d_x, d_y = disk_map(
                    host.offset_scale * z[idx + 4],
                    host.offset_scale * z[idx + 5],
                    xp,
                )
                cos_t = xp.cos(theta)
                sin_t = xp.sin(theta)
                out["host_x"] = out["point0_x"] + r_eff * (
                    d_x * cos_t - q * d_y * sin_t
                )
                out["host_y"] = out["point0_y"] + r_eff * (
                    d_x * sin_t + q * d_y * cos_t
                )
                out["host_offset_rho"] = xp.sqrt(d_x * d_x + d_y * d_y)
            else:
                out["host_x"] = host.center.x_loc + host.center.scale * z[idx + 4]
                out["host_y"] = host.center.y_loc + host.center.scale * z[idx + 5]
        return out

    return Transform(dim=dim, names=tuple(names), physical=physical)
