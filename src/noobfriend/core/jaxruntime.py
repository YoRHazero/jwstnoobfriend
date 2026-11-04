"""Shared JAX runtime setup for noobfriend's NumPyro samplers.

Every JAX-backed sampler in the library configures JAX the same way, through
this module:

- host devices: :func:`configure_host_devices` turns the
  ``NOOB_MCMC_CPU_CORES`` setting into XLA's host-device count **without
  importing jax**. XLA reads that flag once, when its backends start, so the
  call is cheap enough to run at package import time and still takes effect
  as long as nothing has initialized JAX yet;
- platforms: :func:`restrict_to_cpu` keeps a CPU-only sampler's process off
  the GPU entirely, also without importing jax;
- device choice: :func:`resolve_device` turns ``"auto" | "cpu" | "gpu"`` into
  the device a fit runs on, and :func:`resolve_precision` picks its float
  width (``"auto"``: float32 on a GPU, float64 on the CPU);
- precision: :func:`precision` sets the float width inside a block and
  restores it afterwards; :func:`enable_x64` switches JAX to float64 for the
  whole process (for samplers that never run in float32);
- placement: :func:`on_device` / :func:`on_cpu` make a device the default
  inside a block, and :func:`chain_groups` with :func:`chain_method` split
  chains into the groups a device runs side by side (every chain at once,
  vectorized, on a GPU; as many as there are CPU devices, in parallel, on the
  CPU). :func:`chain_map` keeps CPU chains on the CPU even when an
  accelerator is JAX's default backend.

Notes
-----
A ``--xla_force_host_platform_device_count`` already present in ``XLA_FLAGS``
is the user's choice and always wins. When JAX initialized its backends before
:func:`configure_host_devices` ran, the requested count cannot be applied any
more; :func:`cpu_devices` then warns once and callers group their chains by
the device count JAX actually has.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import sys
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any, Literal

from noobfriend.core.env import read_environment

__all__ = [
    "CPU_CORES_VARIABLE",
    "Device",
    "Precision",
    "chain_groups",
    "chain_map",
    "chain_method",
    "configure_host_devices",
    "cpu_devices",
    "enable_x64",
    "on_cpu",
    "on_device",
    "precision",
    "requested_cpu_cores",
    "resolve_device",
    "resolve_precision",
    "restrict_to_cpu",
]

#: A device a fit runs on.
type Device = Literal["cpu", "gpu"]
#: A float width JAX computes in.
type Precision = Literal["float32", "float64"]

#: Environment variable holding the number of CPU cores exposed to JAX.
CPU_CORES_VARIABLE = "NOOB_MCMC_CPU_CORES"
_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
_PLATFORMS_VARIABLE = "JAX_PLATFORMS"
_DEVICE_CHOICES = ("auto", "cpu", "gpu")
_PRECISION_CHOICES = ("auto", "float32", "float64")
#: Distribution names of jaxlib's GPU plugins (``jax-cuda12-plugin``, ...).
_GPU_PLUGIN = re.compile(r"^jax[-_](cuda|rocm)\d*[-_]plugin$", re.IGNORECASE)

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


def restrict_to_cpu() -> None:
    """Keep JAX in this process on the CPU platform only.

    With a CUDA build of jaxlib installed, the first device query starts
    every available backend, even ``jax.devices("cpu")``. A CPU-only process
    then holds a CUDA context for its whole life (about 160 MiB on an RTX
    2080 Ti), and many parallel workers together can leave a concurrent GPU
    job without the memory it needs to start. Restricting the platforms
    before JAX starts its backends avoids touching the GPU at all.

    Rules, in order:

    - ``JAX_PLATFORMS`` already in the environment: the user's choice, left
      untouched;
    - jax not imported yet: set ``JAX_PLATFORMS=cpu`` (does not import jax;
      child processes inherit it);
    - jax imported but its backends not started: set the ``jax_platforms``
      config to ``"cpu"``, unless it already names platforms;
    - backends already started: nothing can change any more; left as is.
    """
    if _PLATFORMS_VARIABLE in os.environ:
        return
    if not _jax_imported():
        os.environ[_PLATFORMS_VARIABLE] = "cpu"
        return
    if _backends_initialized():
        return
    import jax

    if not jax.config.jax_platforms:
        jax.config.update("jax_platforms", "cpu")


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
                "JAX already has. Import the noobfriend sampler package "
                "(inference.spectrum or inference.morphology) before anything "
                "that initializes JAX, or set "
                f"XLA_FLAGS={_DEVICE_COUNT_FLAG}=N yourself.",
                RuntimeWarning,
                stacklevel=2,
            )
    return devices


def resolve_device(device: str) -> Device:
    """Turn a device request into the device a fit runs on.

    Parameters
    ----------
    device
        ``"auto"`` (a GPU when JAX has one, else the CPU), ``"cpu"`` or
        ``"gpu"``.

    Returns
    -------
    {"cpu", "gpu"}
        The resolved device.

    Raises
    ------
    ValueError
        If ``device`` is not one of the three choices.
    RuntimeError
        If ``"gpu"`` is requested and JAX has no GPU backend; the message
        says when the platforms were restricted to the CPU.

    Warns
    -----
    RuntimeWarning
        When ``"auto"`` falls back to the CPU although a GPU plugin is
        installed, because the JAX platforms were restricted (for example a
        ``JAX_PLATFORMS=cpu`` inherited from a spectrum process).

    Notes
    -----
    ``"cpu"`` calls :func:`restrict_to_cpu` first, so a CPU fit started
    before JAX's backends never creates a CUDA context. That restriction
    holds for the rest of the process: a later ``"gpu"`` request in the same
    process fails.
    """
    if device not in _DEVICE_CHOICES:
        raise ValueError(f"device must be one of {_DEVICE_CHOICES}; got {device!r}.")
    if device == "cpu":
        restrict_to_cpu()
        return "cpu"
    if _has_gpu():
        return "gpu"
    restricted = _restricted_platforms()
    if device == "gpu":
        reason = (
            f" because JAX's platforms are restricted to {restricted!r}"
            if restricted
            else ""
        )
        raise RuntimeError(
            f"device='gpu' was requested but JAX has no GPU backend{reason}."
        )
    if restricted and _gpu_plugin_installed():
        warnings.warn(
            "device='auto' runs on the CPU: a GPU plugin is installed but JAX's "
            f"platforms are restricted to {restricted!r} (JAX_PLATFORMS or the "
            "jax_platforms config, e.g. inherited from a spectrum process). "
            "Unset it to use the GPU.",
            RuntimeWarning,
            stacklevel=2,
        )
    return "cpu"


def resolve_precision(precision: str, device: Device) -> Precision:
    """Turn a precision request into the float width a fit computes in.

    Parameters
    ----------
    precision
        ``"auto"`` (float32 on a GPU, float64 on the CPU), ``"float32"`` or
        ``"float64"``.
    device
        The resolved device.

    Returns
    -------
    {"float32", "float64"}
        The resolved precision.

    Raises
    ------
    ValueError
        If ``precision`` is not one of the three choices.
    """
    if precision not in _PRECISION_CHOICES:
        raise ValueError(
            f"precision must be one of {_PRECISION_CHOICES}; got {precision!r}."
        )
    if precision == "auto":
        return "float32" if device == "gpu" else "float64"
    return precision  # type: ignore[return-value]


@contextmanager
def precision(name: Precision) -> Iterator[None]:
    """Compute in ``name`` inside the block; the previous width returns after.

    Everything JAX traces, compiles or creates inside the block uses this
    width, and a function compiled here is compiled again when called under
    another width, so a fit should run entirely inside one block.

    Parameters
    ----------
    name
        ``"float32"`` or ``"float64"``.

    Yields
    ------
    None
        Control returns to the caller with the width in effect.

    Raises
    ------
    ValueError
        If ``name`` is neither width.
    """
    if name not in ("float32", "float64"):
        raise ValueError(f"precision must be 'float32' or 'float64'; got {name!r}.")
    import jax

    with jax.enable_x64(name == "float64"):
        yield


def enable_x64() -> None:
    """Switch JAX to float64 arithmetic for the whole process."""
    import jax

    jax.config.update("jax_enable_x64", True)


@contextmanager
def on_device(device: Device) -> Iterator[None]:
    """Make the first device of a kind JAX's default device inside the block.

    Parameters
    ----------
    device
        ``"cpu"`` or ``"gpu"``.

    Yields
    ------
    None
        Control returns to the caller with that device as default.
    """
    import jax

    first = cpu_devices()[0] if device == "cpu" else jax.devices("gpu")[0]
    with jax.default_device(first):
        yield


@contextmanager
def on_cpu() -> Iterator[None]:
    """Make the first CPU device JAX's default device inside the block.

    Yields
    ------
    None
        Control returns to the caller with the CPU as default device.
    """
    with on_device("cpu"):
        yield


def chain_groups(chains: int, device: Device) -> tuple[int, ...]:
    """Split chains into groups that run one after another.

    Parameters
    ----------
    chains
        Total number of chains.
    device
        ``"gpu"``: one group of every chain (vectorized). ``"cpu"``: groups
        of as many chains as JAX has CPU devices (``NOOB_MCMC_CPU_CORES``).

    Returns
    -------
    tuple of int
        Group sizes, summing to ``chains``.
    """
    if device == "gpu":
        return (chains,)
    width = max(1, min(chains, len(cpu_devices())))
    return tuple(min(width, chains - start) for start in range(0, chains, width))


def chain_method(group_size: int, device: Device) -> str | Callable[..., Any]:
    """Return the NumPyro ``chain_method`` for one chain group.

    Parameters
    ----------
    group_size
        Chains in the group.
    device
        The device the group runs on.

    Returns
    -------
    str or callable
        ``"vectorized"`` on a GPU; on the CPU :func:`chain_map` for several
        chains and ``"sequential"`` for one.
    """
    if device == "gpu":
        return "vectorized"
    return chain_map(group_size) if group_size > 1 else "sequential"


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


def _jax_imported() -> bool:
    """Whether jax is already imported in this process."""
    return "jax" in sys.modules


def _has_gpu() -> bool:
    """Whether JAX has a GPU backend (starts JAX's backends)."""
    import jax

    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


def _restricted_platforms() -> str:
    """Return the platform setting when it excludes every GPU, else ``""``."""
    if _jax_imported():
        import jax

        setting = str(jax.config.jax_platforms or "")
    else:
        setting = os.environ.get(_PLATFORMS_VARIABLE, "")
    names = {name.strip().lower() for name in setting.split(",") if name.strip()}
    return setting if names and not names & {"cuda", "rocm", "gpu"} else ""


def _gpu_plugin_installed() -> bool:
    """Whether a jaxlib GPU plugin distribution is installed."""
    return any(
        _GPU_PLUGIN.match(dist.metadata["Name"] or "")
        for dist in importlib.metadata.distributions()
    )


def _backends_initialized() -> bool:
    """Whether JAX has already started its backends (without importing jax)."""
    bridge = sys.modules.get("jax._src.xla_bridge")
    if bridge is None:
        return False
    probe = getattr(bridge, "backends_are_initialized", None)
    return True if probe is None else bool(probe())
