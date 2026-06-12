"""Component templates: the line-shape priors keyed by ``component`` name.

A :class:`ComponentTemplate` captures what a *kind of line component* implies
regardless of which line it sits on: its emission/absorption ``sign``, the
default free prior support for its velocity FWHM (km/s), and a default initial
width. The built-in registry holds ``narrow`` / ``broad`` / ``absorption``; the
distinction between e.g. ``narrow`` and ``broad`` is exactly the (near-disjoint)
FWHM bounds, which is what lets the two be told apart without label switching.

``component`` is purely the *line-shape type*; a physical role such as an
outflow is expressed by a velocity offset plus an ``abs_fwhm`` override on a
:class:`~noobfriend.inference.spectrum._line.NoobLine`, and named via its
``custom_id`` -- never by inventing a template. New shapes can be added with
:func:`register_template`.

This module is internal; :class:`ComponentTemplate` and :func:`register_template`
are re-exported from :mod:`noobfriend.inference.spectrum`.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ComponentTemplate", "get_template", "register_template", "template_names"]


@dataclass(frozen=True)
class ComponentTemplate:
    """The shape prior implied by a ``component`` name.

    Attributes
    ----------
    name : str
        The ``component`` key (e.g. ``"narrow"``).
    sign : int
        ``+1`` for an emission component, ``-1`` for absorption. The fitted flux
        is always a positive magnitude; this sign decides how it enters the
        model curve.
    fwhm_kms_bounds : tuple of float
        Default ``(lower, upper)`` support for a *free* velocity FWHM (km/s),
        used when the component's width is neither locked nor absolutely
        overridden. ``lower`` is a nominal floor; the fit raises it to the
        instrumental resolution.
    fwhm_kms_init : float
        Default initial velocity FWHM (km/s), chosen inside ``fwhm_kms_bounds``
        and away from its edges.
    """

    name: str
    sign: int
    fwhm_kms_bounds: tuple[float, float]
    fwhm_kms_init: float

    def __post_init__(self) -> None:
        """Validate the sign and bounds.

        Raises
        ------
        ValueError
            If ``sign`` is not ``+1``/``-1``, the bounds are not ``lower < upper``,
            or the init width lies outside the bounds.
        """
        if self.sign not in (1, -1):
            raise ValueError(f"template {self.name!r}: sign must be +1 or -1.")
        lo, hi = self.fwhm_kms_bounds
        if not lo < hi:
            raise ValueError(
                f"template {self.name!r}: fwhm_kms_bounds must have lower < upper, "
                f"got {self.fwhm_kms_bounds}."
            )
        if not lo <= self.fwhm_kms_init <= hi:
            raise ValueError(
                f"template {self.name!r}: fwhm_kms_init {self.fwhm_kms_init} not "
                f"within bounds {self.fwhm_kms_bounds}."
            )


#: Built-in templates. ``narrow`` and ``broad`` are separated by their FWHM
#: bounds (the ``broad`` floor doubles as the narrow/broad discriminator).
_REGISTRY: dict[str, ComponentTemplate] = {
    t.name: t
    for t in (
        ComponentTemplate(
            "narrow", sign=1, fwhm_kms_bounds=(30.0, 1000.0), fwhm_kms_init=250.0
        ),
        ComponentTemplate(
            "broad", sign=1, fwhm_kms_bounds=(1000.0, 10000.0), fwhm_kms_init=2500.0
        ),
        ComponentTemplate(
            "absorption", sign=-1, fwhm_kms_bounds=(50.0, 3000.0), fwhm_kms_init=500.0
        ),
    )
}


def get_template(component: str) -> ComponentTemplate:
    """Return the template registered for a ``component`` name.

    Parameters
    ----------
    component : str
        The component key.

    Returns
    -------
    ComponentTemplate

    Raises
    ------
    ValueError
        If ``component`` is not registered.
    """
    try:
        return _REGISTRY[component]
    except KeyError:
        raise ValueError(
            f"unknown component {component!r}; registered: {template_names()}."
        ) from None


def register_template(template: ComponentTemplate, *, overwrite: bool = False) -> None:
    """Register a custom component template.

    Parameters
    ----------
    template : ComponentTemplate
        The template to add, keyed by its ``name``.
    overwrite : bool, default False
        Allow replacing an existing template of the same name.

    Raises
    ------
    ValueError
        If a template of that name already exists and ``overwrite`` is False.
    """
    if template.name in _REGISTRY and not overwrite:
        raise ValueError(
            f"template {template.name!r} already exists; pass overwrite=True."
        )
    _REGISTRY[template.name] = template


def template_names() -> list[str]:
    """Return the sorted names of all registered templates."""
    return sorted(_REGISTRY)
