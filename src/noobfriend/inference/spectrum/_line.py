"""``NoobLine``: one emission/absorption line at one kinematic component.

A :class:`NoobLine` is a *declaration*, independent of any spectrum: a rest
wavelength, a ``component`` (line-shape type; see
:mod:`noobfriend.inference.spectrum._template`), and how this line's three
parameter axes -- centre velocity, velocity FWHM, and flux -- relate to a
*parent* line. The parent relationship is built with :meth:`NoobLine.derive`,
which returns a new (frozen) line inheriting the parent's identity fields and
tying the requested axes to it. Assembling a list of :class:`NoobLine` and
handing it to :meth:`NoobSpectrum.setup` compiles the tie graph.

Each axis is specified the same way -- ``None`` / a number / a ``(lo, hi)``
tuple -- with an optional absolute override for FWHM and flux:

* centre: ``delta_v_kms`` xor ``delta_wavelength``. ``None`` is the *policy
  default* (a root line floats with a default velocity prior; a derived line
  locks co-moving with its parent); a number is a fixed offset (``0`` pins it);
  a ``(lo, hi)`` tuple is a free bounded offset.
* width: ``lock_fwhm`` (``None`` auto-locks to the parent only when the
  component matches, ``True`` forces the lock, ``False`` frees it) plus
  ``abs_fwhm`` to override with an absolute number / bounded prior (km/s).
* flux: ``flux_ratio`` (``None`` free, a number a fixed ratio of the parent, a
  ``(lo, hi)`` tuple a bounded ratio) plus ``abs_flux`` for an absolute override.

This module is internal; :class:`NoobLine` is re-exported from
:mod:`noobfriend.inference.spectrum`.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import Literal

from noobfriend.inference.spectrum._template import get_template

__all__ = ["NoobLine", "Spec", "classify_spec"]

#: A per-axis user specification: nothing, a fixed value, or ``(lower, upper)``.
Spec = float | int | tuple[float, float] | list[float] | None


def classify_spec(
    value: Spec, where: str
) -> tuple[Literal["none", "fixed", "bounded"], float | tuple[float, float] | None]:
    """Classify and validate one axis specification.

    Parameters
    ----------
    value : float or tuple of float or None
        ``None`` (unset), a scalar (a fixed value), or a length-2 ``(lower,
        upper)`` sequence (a bounded prior).
    where : str
        Context for error messages.

    Returns
    -------
    kind : {"none", "fixed", "bounded"}
        Which form ``value`` took.
    payload : float or tuple of float or None
        The float for ``"fixed"``, the ``(lower, upper)`` tuple for
        ``"bounded"``, or ``None`` for ``"none"``.

    Raises
    ------
    ValueError
        If ``value`` is neither a scalar nor a length-2 ``lower < upper`` pair.
    """
    if value is None:
        return "none", None
    if isinstance(value, bool):
        raise ValueError(f"{where}: bool is not a valid value, got {value!r}.")
    if isinstance(value, (int, float)):
        return "fixed", float(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        if not lo < hi:
            raise ValueError(
                f"{where}: a (lower, upper) bound needs lower < upper, got {value!r}."
            )
        return "bounded", (lo, hi)
    raise ValueError(
        f"{where}: expected None, a number, or a (lower, upper) pair, got {value!r}."
    )


@dataclass(frozen=True, repr=False)
class NoobLine:
    """A single line/component declaration and its tie to a parent.

    Construct a *root* line directly; derive related lines with
    :meth:`derive`. Frozen: every relationship is a new instance.

    Parameters
    ----------
    linename : str
        Line name (e.g. ``"Halpha"``); lines sharing a name are grouped when ids
        are generated.
    rest_wavelength : float
        Rest-frame wavelength in ``unit``. Required for a root line; inherited by
        :meth:`derive` when omitted.
    unit : str
        Wavelength unit of ``rest_wavelength`` (e.g. ``"Angstrom"``). Must match
        the spectrum's ``wave_unit`` at setup. Required for a root line.
    component : str
        Line-shape type; one of the registered templates (``narrow`` / ``broad``
        / ``absorption`` by default). Sets the sign and default width prior, and
        drives the FWHM auto-lock.
    custom_id : str, optional
        Overrides the id generated at setup, so any name (e.g. ``"Halpha_outflow"``)
        is possible without overloading ``component``.
    delta_v_kms, delta_wavelength : float or tuple of float, optional
        The centre-velocity offset, in km/s or in ``unit`` respectively (mutually
        exclusive). See the module docstring for the ``None`` / number / tuple
        semantics.
    lock_fwhm : bool, optional
        Force (``True``) or forbid (``False``) locking this width to the parent's;
        ``None`` auto-locks only when ``component`` matches the parent's.
    abs_fwhm : float or tuple of float, optional
        Absolute velocity-FWHM override (km/s): a number fixes it, a ``(lo, hi)``
        tuple bounds it. Overrides the template default and the auto-lock.
    flux_ratio : float or tuple of float, optional
        Flux relative to the parent: ``None`` free, a number a fixed ratio, a
        ``(lo, hi)`` tuple a bounded ratio.
    abs_flux : float or tuple of float, optional
        Absolute flux override: a number fixes it, a ``(lo, hi)`` tuple bounds it.
    parent : NoobLine, optional
        The line this one was derived from. Set by :meth:`derive`; leave unset to
        build a root line.
    """

    linename: str
    _: KW_ONLY
    rest_wavelength: float | None = None
    unit: str | None = None
    component: str | None = None
    custom_id: str | None = None
    delta_v_kms: Spec = None
    delta_wavelength: Spec = None
    lock_fwhm: bool | None = None
    abs_fwhm: Spec = None
    flux_ratio: Spec = None
    abs_flux: Spec = None
    parent: NoobLine | None = None

    def __post_init__(self) -> None:
        """Validate identity, axis specs, and per-axis conflicts.

        Raises
        ------
        ValueError
            On a missing/invalid identity field, an unknown ``component``, a
            malformed axis spec, a per-axis conflict (both centre specs, a forced
            FWHM lock with ``abs_fwhm``, or ``flux_ratio`` with ``abs_flux``), or a
            parent-only option (``lock_fwhm=True`` / ``flux_ratio``) on a root line.
        """
        if not self.linename:
            raise ValueError("linename must be a non-empty string.")
        if self.rest_wavelength is None or self.rest_wavelength <= 0:
            raise ValueError(
                f"{self.linename}: rest_wavelength must be a positive number."
            )
        if not self.unit:
            raise ValueError(f"{self.linename}: unit is required (e.g. 'Angstrom').")
        if self.component is None:
            raise ValueError(f"{self.linename}: component is required.")
        get_template(self.component)  # validates the component name

        if self.lock_fwhm is not None and not isinstance(self.lock_fwhm, bool):
            raise ValueError(
                f"{self.linename}: lock_fwhm must be True, False, or None."
            )

        classify_spec(self.delta_v_kms, f"{self.linename}.delta_v_kms")
        classify_spec(self.delta_wavelength, f"{self.linename}.delta_wavelength")
        classify_spec(self.abs_fwhm, f"{self.linename}.abs_fwhm")
        classify_spec(self.flux_ratio, f"{self.linename}.flux_ratio")
        classify_spec(self.abs_flux, f"{self.linename}.abs_flux")

        if self.delta_v_kms is not None and self.delta_wavelength is not None:
            raise ValueError(
                f"{self.linename}: give delta_v_kms or delta_wavelength, not both."
            )
        if self.lock_fwhm is True and self.abs_fwhm is not None:
            raise ValueError(
                f"{self.linename}: lock_fwhm=True conflicts with abs_fwhm "
                "(absolute override vs lock to parent)."
            )
        if self.flux_ratio is not None and self.abs_flux is not None:
            raise ValueError(f"{self.linename}: give flux_ratio or abs_flux, not both.")

        if self.parent is None:
            if self.lock_fwhm is True:
                raise ValueError(
                    f"{self.linename}: lock_fwhm=True needs a parent; this is a "
                    "root line. Derive it with .derive() to tie a width."
                )
            if self.flux_ratio is not None:
                raise ValueError(
                    f"{self.linename}: flux_ratio needs a parent; this is a root "
                    "line. Use abs_flux for an absolute prior."
                )

    @property
    def is_root(self) -> bool:
        """Whether this line was built directly (no derive parent)."""
        return self.parent is None

    def derive(
        self,
        linename: str | None = None,
        *,
        rest_wavelength: float | None = None,
        unit: str | None = None,
        component: str | None = None,
        custom_id: str | None = None,
        delta_v_kms: Spec = None,
        delta_wavelength: Spec = None,
        lock_fwhm: bool | None = None,
        abs_fwhm: Spec = None,
        flux_ratio: Spec = None,
        abs_flux: Spec = None,
    ) -> NoobLine:
        """Derive a related line tied to this one.

        Identity fields (``linename``, ``rest_wavelength``, ``unit``,
        ``component``) are inherited from this line when omitted; the
        relationship fields describe the new line's tie to *this* line (its
        parent) and reset each call. All relative ties (``delta_*``,
        ``flux_ratio``, the FWHM auto-lock) reference this parent.

        Parameters
        ----------
        linename : str, optional
            New line name; defaults to the parent's.
        rest_wavelength, unit, component : optional
            Inherited from the parent when omitted.
        custom_id : str, optional
            Id override for the derived line.
        delta_v_kms, delta_wavelength, lock_fwhm, abs_fwhm, flux_ratio, abs_flux
            Tie / override specs for the new line; see :class:`NoobLine`.

        Returns
        -------
        NoobLine
            The new line with ``parent`` set to this one.
        """
        return NoobLine(
            linename=self.linename if linename is None else linename,
            rest_wavelength=(
                self.rest_wavelength if rest_wavelength is None else rest_wavelength
            ),
            unit=self.unit if unit is None else unit,
            component=self.component if component is None else component,
            custom_id=custom_id,
            delta_v_kms=delta_v_kms,
            delta_wavelength=delta_wavelength,
            lock_fwhm=lock_fwhm,
            abs_fwhm=abs_fwhm,
            flux_ratio=flux_ratio,
            abs_flux=abs_flux,
            parent=self,
        )

    def __repr__(self) -> str:
        """Concise representation that does not recurse into ``parent``."""
        bits = [
            repr(self.linename),
            f"rest={self.rest_wavelength}{self.unit or ''}",
            f"component={self.component!r}",
        ]
        if self.custom_id is not None:
            bits.append(f"custom_id={self.custom_id!r}")
        if self.parent is not None:
            bits.append(f"from={self.parent.linename}.{self.parent.component}")
        return f"NoobLine({', '.join(bits)})"
