"""Model family for AGN/host spatial decomposition.

The model space is a small fixed family: ``n_points`` point sources plus an
optional single Sersic host. Model *presence* questions ("is there a host?",
"is this a dual AGN?") are answered by fitting several members of the family
and comparing them predictively, never by letting an amplitude hit a boundary
inside one model.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ModelSpec"]


@dataclass(frozen=True)
class ModelSpec:
    """One member of the decomposition model family.

    Parameters
    ----------
    n_points : int
        Number of point-source components. May be zero only when
        ``has_host`` is true.
    has_host : bool
        Whether the model includes a single Sersic host component. When
        ``n_points >= 1`` the host centre is constrained so the first point
        source lies inside the host's one-``r_eff`` ellipse.

    Examples
    --------
    >>> ModelSpec.parse("point")
    ModelSpec(n_points=1, has_host=False)
    >>> ModelSpec.parse("2point+host").name
    '2point+host'
    """

    n_points: int
    has_host: bool

    def __post_init__(self) -> None:
        """Validate the component counts."""
        if self.n_points < 0:
            raise ValueError("ModelSpec.n_points must be >= 0.")
        if self.n_points == 0 and not self.has_host:
            raise ValueError("ModelSpec needs at least one component.")

    @property
    def name(self) -> str:
        """Return the canonical model name, e.g. ``"2point+host"``."""
        if self.n_points == 0:
            return "host"
        prefix = "point" if self.n_points == 1 else f"{self.n_points}point"
        return f"{prefix}+host" if self.has_host else prefix

    @property
    def component_names(self) -> tuple[str, ...]:
        """Return per-source component names in amplitude order."""
        points = tuple(f"point{k}" for k in range(self.n_points))
        return points + (("host",) if self.has_host else ())

    @property
    def n_amplitudes(self) -> int:
        """Return the number of linear amplitudes per band (incl. background)."""
        return self.n_points + int(self.has_host) + 1

    @classmethod
    def parse(cls, model: "ModelSpec | str") -> "ModelSpec":
        """Parse a model name like ``"point"``, ``"host"`` or ``"2point+host"``.

        Parameters
        ----------
        model : ModelSpec or str
            An existing spec (returned unchanged) or a canonical name.

        Returns
        -------
        ModelSpec
            The parsed specification.

        Raises
        ------
        ValueError
            If the name is not a member of the model family.
        """
        if isinstance(model, cls):
            return model
        name = model.strip()
        if name == "host":
            return cls(n_points=0, has_host=True)
        has_host = name.endswith("+host")
        core = name.removesuffix("+host")
        if core.endswith("point"):
            count = core.removesuffix("point")
            if count == "" or count.isdigit():
                return cls(n_points=int(count) if count else 1, has_host=has_host)
        raise ValueError(
            f"unknown model {model!r}; expected e.g. 'point', 'host', "
            "'point+host', '2point+host'."
        )
