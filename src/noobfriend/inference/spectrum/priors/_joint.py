"""Joint multivariate-normal priors carried over from an earlier fit.

A :class:`JointNormal` is built by an MCMC result's ``as_prior`` and is never
constructed by users. It holds the posterior mean and covariance of several
parameters in unconstrained coordinates (``log`` for positive parameters,
``logit`` for fractions, identity otherwise); each parameter of the new
declarations receives a :class:`JointMember` pointing back at it. ``prepare``
gathers the members sharing one joint prior and marginalizes the normal onto
the members actually present.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from noobfriend.inference.spectrum._xp import readonly
from noobfriend.inference.spectrum.priors._domain import Domain
from noobfriend.inference.spectrum.priors._priors import Prior


@dataclass(frozen=True, slots=True)
class PriorSource:
    """One frame of the fit a joint prior was learned from.

    Attributes
    ----------
    digest
        Fingerprint of the frame's arrays.
    fitted
        Boolean mask of the native pixels that fit used.
    """

    digest: str
    fitted: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class JointNormal:
    """Multivariate normal prior in unconstrained coordinates.

    Attributes
    ----------
    names
        Flat parameter names in the fit the prior was learned from.
    domains
        Each parameter's domain; fixes its unconstrained transform.
    mean, covariance
        Moments in unconstrained coordinates.
    sources
        The frames of that fit, for the data-reuse check.
    """

    names: tuple[str, ...]
    domains: tuple[Domain, ...]
    mean: np.ndarray
    covariance: np.ndarray
    sources: tuple[PriorSource, ...] = field(default=())

    def __post_init__(self) -> None:
        """Validate shapes and positive definiteness."""
        size = len(self.names)
        if len(set(self.names)) != size or len(self.domains) != size:
            raise ValueError("JointNormal names must be unique, one domain each.")
        mean = readonly(self.mean)
        covariance = readonly(self.covariance)
        if mean.shape != (size,) or covariance.shape != (size, size):
            raise ValueError("JointNormal mean/covariance shapes do not match names.")
        if not np.allclose(covariance, covariance.T):
            raise ValueError("JointNormal covariance must be symmetric.")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            raise ValueError(
                "JointNormal covariance must be positive definite."
            ) from None
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    def index(self, name: str) -> int:
        """Return the position of ``name``."""
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"{name!r} is not part of this joint prior.") from None

    def member(self, name: str) -> JointMember:
        """Return the prior handle for one parameter."""
        self.index(name)
        return JointMember(self, name)

    def marginal(self, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return mean and covariance of the normal marginalized onto ``names``.

        Parameters
        ----------
        names
            Members to keep, in the order wanted.

        Returns
        -------
        mean, covariance : numpy.ndarray
            The sub-vector and sub-matrix (exact for a normal).
        """
        indices = [self.index(name) for name in names]
        return (
            readonly(self.mean[indices]),
            readonly(self.covariance[np.ix_(indices, indices)]),
        )


@dataclass(frozen=True, slots=True)
class JointMember(Prior):
    """One parameter's share of a :class:`JointNormal`.

    Attributes
    ----------
    joint
        The shared prior.
    name
        The member's flat name in the fit the prior came from.
    """

    joint: JointNormal
    name: str

    def resolve(self, domain: Domain) -> JointMember:
        """Check the parameter's domain matches the member's."""
        expected = self.joint.domains[self.joint.index(self.name)]
        if expected.transform != domain.transform:
            raise ValueError(
                f"joint prior member {self.name!r} was learned on a {expected.name} "
                f"parameter and cannot describe a {domain.name} one."
            )
        return self

    def bounds(self) -> tuple[float, float]:
        """Return the member's domain."""
        domain = self.joint.domains[self.joint.index(self.name)]
        return domain.low, domain.high

    @property
    def is_proper(self) -> bool:
        """Always ``True``."""
        return True

    def scaled(self, factor: float) -> JointMember:
        """Unavailable: a joint member cannot change units."""
        raise ValueError(
            f"joint prior member {self.name!r} cannot be rescaled; declare the "
            "line in the spectrum's wavelength unit."
        )

    def log_density(self, xp: Any, value: Any) -> Any:
        """Unavailable: members are only sampled jointly."""
        raise NotImplementedError("joint prior members have no separate density.")

    def __str__(self) -> str:
        """Compact form naming the joint prior's size."""
        return f"JointNormal[{self.name}; {len(self.joint.names)} params]"
