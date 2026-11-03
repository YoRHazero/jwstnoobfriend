"""Shared JAX runtime setup for noobfriend's NumPyro samplers.

Every JAX-backed sampler in the library configures JAX the same way, through
this module:

- host devices: :func:`configure_host_devices` turns the
  ``NOOB_MCMC_CPU_CORES`` setting into XLA's host-device count **without
  importing jax**. XLA reads that flag once, when its backends start, so the
  call is cheap enough to run at package import time and still takes effect
  as long as nothing has initialized JAX yet;
- precision: :func:`enable_x64` switches JAX to float64;
- placement: :func:`cpu_devices`, :func:`on_cpu` and :func:`chain_map` keep
  sampling on the CPU even when an accelerator backend is installed.

Notes
-----
A ``--xla_force_host_platform_device_count`` already present in ``XLA_FLAGS``
is the user's choice and always wins. When JAX initialized its backends before
:func:`configure_host_devices` ran, the requested count cannot be applied any
more; :func:`cpu_devices` then warns once and callers group their chains by
the device count JAX actually has.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any

from noobfriend.core.env import read_environment

__all__ = [
    "CPU_CORES_VARIABLE",
    "chain_map",
    "configure_host_devices",
    "cpu_devices",
    "enable_x64",
    "on_cpu",
    "requested_cpu_cores",
]

#: Environment variable holding the number of CPU cores exposed to JAX.
CPU_CORES_VARIABLE = "NOOB_MCMC_CPU_CORES"
_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"

#: ``True`` when JAX was already running at configuration time, so the
#: requested device count could not be applied.
_configured_too_late = False
_mismatch_warned = False


def requested_cpu_cores() -> int:
    """Return the number of CPU cores the user asked JAX to expose.

    Reads ``NOOB_MCMC_CPU_CORES`` from :data:`os.environ`, falling back to the
    layered ``.env`` files (read without modifying the environment).

    Returns
    -------
    int
        The configured core count, or every core of this machine when the
        variable is unset or empty.

    Raises
    ------
    ValueError
        If the variable is set to anything other than a positive integer.
    """
    raw = os.environ.get(CPU_CORES_VARIABLE)
    if raw is None:
        raw = read_environment().get(CPU_CORES_VARIABLE)
    if raw is None or not raw.strip():
        return os.cpu_count() or 1
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{CPU_CORES_VARIABLE} must be a positive integer; got {raw!r}."
        ) from None
    if value < 1:
        raise ValueError(
            f"{CPU_CORES_VARIABLE} must be a positive integer; got {raw!r}."
        )
    return value


def configure_host_devices() -> None:
    """Write the requested CPU device count into ``XLA_FLAGS``.

    Does not import jax. Leaves ``XLA_FLAGS`` untouched when it already sets
    the host-device count (the user's choice wins) or when JAX's backends are
    already initialized (the flag would be ignored); in the latter case
    :func:`cpu_devices` warns once if the counts differ.

    Raises
    ------
    ValueError
        If ``NOOB_MCMC_CPU_CORES`` is not a positive integer.
    """
    global _configured_too_late
    flags = os.environ.get("XLA_FLAGS", "")
    if _DEVICE_COUNT_FLAG in flags:
        return
    if _backends_initialized():
        _configured_too_late = True
        return
    count = requested_cpu_cores()
    os.environ["XLA_FLAGS"] = f"{flags} {_DEVICE_COUNT_FLAG}={count}".strip()


def cpu_devices() -> list[Any]:
    """Return JAX's CPU devices, warning once about an unapplied core count.

    Returns
    -------
    list of jax.Device
        The CPU devices, in JAX's order.

    Warns
    -----
    RuntimeWarning
        Once per process, when JAX was initialized before
        :func:`configure_host_devices` could apply ``NOOB_MCMC_CPU_CORES`` and
        the actual device count differs from the requested one.
    """
    global _mismatch_warned
    import jax

    devices = jax.devices("cpu")
    if _configured_too_late and not _mismatch_warned:
        _mismatch_warned = True
        requested = requested_cpu_cores()
        if len(devices) != requested:
            warnings.warn(
                f"JAX was initialized before noobfriend could expose {requested} "
                f"CPU devices; chains are grouped by the {len(devices)} device(s) "
                "JAX already has. Import noobfriend.inference.spectrum before "
                "anything that initializes JAX, or set "
                f"XLA_FLAGS={_DEVICE_COUNT_FLAG}=N yourself.",
                RuntimeWarning,
                stacklevel=2,
            )
    return devices


def enable_x64() -> None:
    """Switch JAX to float64 arithmetic."""
    import jax

    jax.config.update("jax_enable_x64", True)


@contextmanager
def on_cpu() -> Iterator[None]:
    """Make the first CPU device JAX's default device inside the block.

    Yields
    ------
    None
        Control returns to the caller with the CPU as default device.
    """
    import jax

    with jax.default_device(cpu_devices()[0]):
        yield


def chain_map(n_devices: int) -> str | Callable[..., Any]:
    """Return a NumPyro ``chain_method`` running chains on CPU devices.

    Parameters
    ----------
    n_devices
        Number of chains run side by side, one per CPU device.

    Returns
    -------
    str or callable
        ``"parallel"`` when the CPU is JAX's default backend; otherwise a
        :func:`jax.pmap` pinned to the first ``n_devices`` CPU devices, since
        NumPyro's own ``"parallel"`` maps over the default (accelerator)
        backend.
    """
    import jax

    if jax.default_backend() == "cpu":
        return "parallel"
    return partial(jax.pmap, devices=cpu_devices()[:n_devices])


def _backends_initialized() -> bool:
    """Whether JAX has already started its backends (without importing jax)."""
    bridge = sys.modules.get("jax._src.xla_bridge")
    if bridge is None:
        return False
    probe = getattr(bridge, "backends_are_initialized", None)
    return True if probe is None else bool(probe())
