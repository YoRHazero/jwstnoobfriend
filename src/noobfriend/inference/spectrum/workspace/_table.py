"""The workspace's single parameter table.

Every parameter of the fit has one row, named by its flat dotted path
(``Ha_broad.fwhm``, ``Ha_broad.laplace.fwhm``, ``continuum.c0``,
``continuum.c0.spread``, ``frame.v1.continuum.c0``, ``frame.v1.noise.rho``).
That name is used everywhere: sampling sites, results and saved files. MLE,
MCMC, results, ``as_prior`` and the summary all read this table.

Rows describe the *declared* parameters. The physical quantities every line
reports (``center``, ``delta_v``, ``fwhm``, ``flux`` and kernel parameters)
are :class:`Quantity` expressions over those rows.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from noobfriend.inference.spectrum.priors import Domain, JointMember, JointNormal, Prior
from noobfriend.inference.spectrum.workspace._expressions import Expression

type ParameterKind = Literal["free", "fixed", "pooled"]


@dataclass(frozen=True, slots=True)
class Parameter:
    """One declared parameter.

    Attributes
    ----------
    name
        Flat dotted name.
    owner
        Component (line name, ``"continuum"``) or ``"frame.<id>"``.
    role
        The parameter's role within its owner (``"fwhm"``,
        ``"fwhm_ratio_to_base"``, ``"laplace.fraction"``, ``"c1"``,
        ``"noise.rho"``, ...).
    unit
        Unit text (``"km/s"``, ``"angstrom"``, ``"flux*angstrom"``, ``""``).
    domain
        The values the parameter can take.
    kind
        ``"free"`` (has a prior), ``"fixed"`` (has a value) or ``"pooled"``
        (a frame's value drawn around a pooled mean).
    prior
        The resolved prior of a free parameter (a ``JointMember`` for joint
        prior members).
    value
        The value of a fixed parameter.
    pooled_from
        For a pooled row: the mean's name; the spread is ``<mean>.spread``.
    log_pooled
        Whether pooling happens in log space (power-law level).
    """

    name: str
    owner: str
    role: str
    unit: str
    domain: Domain
    kind: ParameterKind
    prior: Prior | None = None
    value: float | None = None
    pooled_from: str | None = None
    log_pooled: bool = False

    @property
    def is_free(self) -> bool:
        """Whether the parameter has its own prior."""
        return self.kind == "free"

    @property
    def is_joint(self) -> bool:
        """Whether the prior is a share of a joint prior."""
        return isinstance(self.prior, JointMember)

    def rule(self) -> str:
        """Describe how the parameter is set, for the summary."""
        if self.kind == "fixed":
            return f"fixed {self.value:.6g}"
        if self.kind == "pooled":
            space = "log " if self.log_pooled else ""
            return f"pooled: {space}{self.pooled_from} +- {self.pooled_from}.spread"
        return str(self.prior)


@dataclass(frozen=True, slots=True)
class Quantity:
    """A physical quantity reported for a line.

    Attributes
    ----------
    name
        Flat name, e.g. ``"NII.fwhm"``.
    owner
        Line name.
    role
        ``"center"``, ``"delta_v"``, ``"fwhm"``, ``"flux"``, ``"<kind>.fwhm"``
        or ``"<kind>.fraction"``.
    unit
        Unit text.
    expression
        How it follows from the declared parameters (for text).
    evaluator
        The same with constants folded (for evaluation).
    """

    name: str
    owner: str
    role: str
    unit: str
    expression: Expression
    evaluator: Expression

    @property
    def is_direct(self) -> bool:
        """Whether the quantity is itself one declared parameter."""
        return self.expression.references() == (self.name,)


@dataclass(frozen=True, slots=True, eq=False)
class JointBlock:
    """The part of one joint prior used by this workspace.

    Attributes
    ----------
    joint
        The full joint prior.
    parameters
        This workspace's parameter names, one per used member.
    members
        The matching member names of the joint prior.
    mean, covariance
        The joint prior marginalized onto ``members`` (unconstrained
        coordinates).
    marginalized
        Members not used here (absent or overridden), integrated out exactly.
    """

    joint: JointNormal
    parameters: tuple[str, ...]
    members: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray
    marginalized: tuple[str, ...]

    @property
    def domains(self) -> tuple[Domain, ...]:
        """Domains of the used members."""
        return tuple(
            self.joint.domains[self.joint.index(name)] for name in self.members
        )


class ParameterTable(Mapping[str, Parameter]):
    """Ordered rows by flat name plus the line quantities.

    Parameters
    ----------
    rows
        Parameters in declaration order.
    quantities
        Physical line quantities in declaration order.
    """

    def __init__(self, rows: list[Parameter], quantities: list[Quantity]) -> None:
        self._rows = {row.name: row for row in rows}
        if len(self._rows) != len(rows):
            raise ValueError("duplicate parameter names in the table.")
        self._quantities = {quantity.name: quantity for quantity in quantities}

    def __getitem__(self, name: str) -> Parameter:
        """Return one row."""
        try:
            return self._rows[name]
        except KeyError:
            raise KeyError(f"no parameter {name!r} in this workspace.") from None

    def __iter__(self) -> Iterator[str]:
        """Iterate over names in declaration order."""
        return iter(self._rows)

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self._rows)

    @property
    def quantities(self) -> Mapping[str, Quantity]:
        """Physical line quantities by flat name."""
        return self._quantities

    def free(self) -> tuple[Parameter, ...]:
        """Rows with their own prior, in order."""
        return tuple(row for row in self._rows.values() if row.kind == "free")

    def pooled(self) -> tuple[Parameter, ...]:
        """Pooled per-frame rows, in order."""
        return tuple(row for row in self._rows.values() if row.kind == "pooled")

    def owned_by(self, owner: str) -> tuple[Parameter, ...]:
        """Rows of one owner, in order."""
        return tuple(row for row in self._rows.values() if row.owner == owner)
