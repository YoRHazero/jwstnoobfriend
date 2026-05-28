"""Progress-bar helpers built on :mod:`rich.progress` and the shared console.

Three coordinated entry points:

- :func:`track` — re-export of :func:`rich.progress.track` for inline use
  inside a ``for`` loop, with the shared noobfriend console injected as the
  default.
- :func:`track_iter` — decorator form: wraps an iterable function argument
  with a determinate progress bar.
- :func:`track_time` — decorator form: shows a spinner and elapsed-time
  counter while a function runs (no iterable to track).
"""

import functools
import inspect
from collections.abc import Callable, Iterable, Iterator, Sized
from typing import Any, ParamSpec, TypeVar

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.progress import (
    track as _rich_track,
)

from noobfriend.core.console import console

P = ParamSpec("P")
R = TypeVar("R")

_ITER_COLUMNS: tuple[ProgressColumn, ...] = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)

_TIMER_COLUMNS: tuple[ProgressColumn, ...] = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    TimeElapsedColumn(),
)


def _build(
    func: Callable[P, R],
    *,
    columns: tuple[ProgressColumn, ...],
    description: str,
    refresh_per_second: int,
    iter_param: str | None,
    auto_detect: bool,
) -> Callable[P, R]:
    """Build a progress-aware wrapper for ``func`` shared by the public decorators.

    Parameters
    ----------
    func : callable
        Function to wrap.
    columns : tuple of ProgressColumn
        Columns rendered in the live progress display.
    description : str
        Description text shown next to the spinner.
    refresh_per_second : int
        Refresh rate of the live display, in Hz.
    iter_param : str or None
        Name of the function parameter whose iterable should be tracked. If
        ``None`` *and* ``auto_detect`` is ``True``, the first non-string
        :class:`~collections.abc.Iterable` argument is located at call time.
        If ``None`` *and* ``auto_detect`` is ``False``, no iterable is tracked
        and only the spinner / elapsed counter are shown.
    auto_detect : bool
        Whether to auto-detect the first iterable argument when ``iter_param``
        is ``None``.

    Returns
    -------
    callable
        Wrapped function preserving the original signature.
    """
    sig = inspect.signature(func) if (iter_param is not None or auto_detect) else None

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with Progress(
            *columns, console=console, refresh_per_second=refresh_per_second
        ) as progress:
            if sig is None:
                progress.add_task(description, total=None)
                return func(*args, **kwargs)

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            key = iter_param
            if key is None:
                key = next(
                    (
                        k
                        for k, v in bound.arguments.items()
                        if isinstance(v, Iterable)
                        and not isinstance(v, (str, bytes, bytearray))
                    ),
                    None,
                )
                if key is None:
                    raise ValueError(
                        "no iterable parameter found in function arguments"
                    )
            if key not in bound.arguments:
                raise ValueError(f"parameter {key!r} not found in function arguments")

            iterable = bound.arguments[key]
            total = len(iterable) if isinstance(iterable, Sized) else None
            task = progress.add_task(description, total=total)

            def tracked() -> Iterator[Any]:
                for item in iterable:
                    yield item
                    progress.advance(task)

            bound.arguments[key] = tracked()
            return func(*bound.args, **bound.kwargs)

    return wrapper


def track(
    sequence: Iterable[Any],
    description: str = "Working...",
    **kwargs: Any,
) -> Iterable[Any]:
    """Wrap ``sequence`` with a progress bar; re-export of :func:`rich.progress.track`.

    The shared :data:`noobfriend.core.console.console` is injected as the
    default ``console`` keyword so the progress bar cooperates with the
    package logging handler (no interleaved redraws).

    Parameters
    ----------
    sequence : iterable
        The iterable to wrap. ``len(sequence)`` is used for the total when
        available; otherwise the progress shows an indeterminate spinner.
    description : str, default ``"Working..."``
        Description text shown next to the bar.
    **kwargs
        Forwarded to :func:`rich.progress.track`. Passing ``console=`` here
        overrides the shared console.

    Returns
    -------
    iterable
        Iterable yielding the items of ``sequence`` while advancing the bar.

    Examples
    --------
    >>> for item in track([1, 2, 3], description="Counting"):  # doctest: +SKIP
    ...     ...
    """
    kwargs.setdefault("console", console)
    return _rich_track(sequence, description, **kwargs)


def track_iter(
    iter_param: str | None = None,
    description: str = "Processing...",
    refresh_per_second: int = 10,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function to display a determinate progress bar over one of its arguments.

    Parameters
    ----------
    iter_param : str or None, default ``None``
        Name of the function parameter to track. If ``None``, the first
        function argument that is a non-string
        :class:`~collections.abc.Iterable` is auto-detected at call time.
    description : str, default ``"Processing..."``
        Description text shown next to the bar.
    refresh_per_second : int, default ``10``
        Refresh rate of the live display, in Hz.

    Returns
    -------
    callable
        A decorator that wraps a callable so its iterable argument is
        consumed through a progress-tracked generator.

    Examples
    --------
    >>> @track_iter("files")  # doctest: +SKIP
    ... def reduce(files):
    ...     for f in files:
    ...         ...
    """

    def _decorator(func: Callable[P, R]) -> Callable[P, R]:
        return _build(
            func,
            columns=_ITER_COLUMNS,
            description=description,
            refresh_per_second=refresh_per_second,
            iter_param=iter_param,
            auto_detect=(iter_param is None),
        )

    return _decorator


def track_time(
    description: str = "Running...",
    refresh_per_second: int = 4,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function to display a spinner with elapsed time while it runs.

    Use when the function does not naturally iterate over a parameter and you
    just want a live "still working" indicator with a running clock.

    Parameters
    ----------
    description : str, default ``"Running..."``
        Description text shown next to the spinner.
    refresh_per_second : int, default ``4``
        Refresh rate of the live display, in Hz.

    Returns
    -------
    callable
        A decorator that wraps a callable so the spinner and elapsed-time
        column are shown for the duration of the call.

    Examples
    --------
    >>> @track_time("Fitting model")  # doctest: +SKIP
    ... def fit():
    ...     ...
    """

    def _decorator(func: Callable[P, R]) -> Callable[P, R]:
        return _build(
            func,
            columns=_TIMER_COLUMNS,
            description=description,
            refresh_per_second=refresh_per_second,
            iter_param=None,
            auto_detect=False,
        )

    return _decorator
