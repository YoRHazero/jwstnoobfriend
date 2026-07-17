"""Resolve spectrum data and line declarations into one fit workspace."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose
from typing import TYPE_CHECKING, Literal

from noobfriend.inference.spectrum.data.axis import _format_center_label
from noobfriend.inference.spectrum.line import NoobLine
from noobfriend.inference.spectrum.types import convert_wavelength
from noobfriend.inference.spectrum.workspace.continuum import (
    ContinuumSpec,
    build_continuum,
)
from noobfriend.inference.spectrum.workspace.handles import LineHandle

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.data import NoobSpectrum
    from noobfriend.inference.spectrum.model import NoobSpectrumModel
    from noobfriend.inference.spectrum.workspace.mle import MLEFitResult

type _LineEntry = tuple[str | None, NoobLine]
_RESERVED_LINE_IDS = frozenset({"continuum"})


@dataclass(frozen=True, slots=True)
class NoobFitWorkspace:
    """A prepared spectrum plus one resolved line combination."""

    spectrum: NoobSpectrum
    handles: tuple[LineHandle, ...]
    id_mode: Literal["auto", "manual"]
    continuum: ContinuumSpec

    @classmethod
    def build(
        cls,
        spectrum: NoobSpectrum,
        lines: Sequence[NoobLine] | Mapping[str, NoobLine],
        *,
        continuum_order: int = 1,
        continuum_lambda_0: float | None = None,
    ) -> NoobFitWorkspace:
        """Build a workspace from either auto-id or manual-id input."""
        id_mode, entries = _normalize_line_entries(lines)
        _validate_line_objects(entries)
        _validate_line_graph(entries)
        handles = _build_handles(spectrum, entries, id_mode=id_mode)
        continuum = build_continuum(
            handles,
            continuum_order=continuum_order,
            continuum_lambda_0=continuum_lambda_0,
        )
        return cls(
            spectrum=spectrum, handles=handles, id_mode=id_mode, continuum=continuum
        )

    @property
    def lines(self) -> tuple[NoobLine, ...]:
        """Original line objects in workspace order."""
        return tuple(handle.line for handle in self.handles)

    @property
    def ids(self) -> tuple[str, ...]:
        """Workspace-local line ids in order."""
        return tuple(handle.id for handle in self.handles)

    @property
    def roots(self) -> tuple[LineHandle, ...]:
        """Handles whose line has no base."""
        return tuple(handle for handle in self.handles if handle.line.base is None)

    @property
    def derived(self) -> tuple[LineHandle, ...]:
        """Handles whose line was derived from another line."""
        return tuple(handle for handle in self.handles if handle.line.base is not None)

    def handle_for(self, line: NoobLine) -> LineHandle:
        """Return the handle for ``line`` by object identity."""
        for handle in self.handles:
            if handle.line is line:
                return handle
        raise KeyError("line is not part of this workspace.")

    def model(self) -> NoobSpectrumModel:
        """Build and return one persistent spectrum model ready for sampling.

        The model uses only the spectrum data and ``NoobLine`` declarations;
        no MLE point is computed or supplied as an initial value.
        """
        from noobfriend.inference.spectrum.model import NoobSpectrumModel

        return NoobSpectrumModel.from_workspace(self)

    def mle(
        self,
        *,
        absorption_bound_multipliers: Sequence[float] = (0.5, 1.0, 2.0, 5.0),
        cancellation_threshold: float | None = 0.37,
        relative_likelihood_min: float = 0.5,
        random_seed: int = 1729,
    ) -> MLEFitResult:
        """Fit this workspace with the built-in robust maximum-likelihood workflow.

        Parameters
        ----------
        absorption_bound_multipliers
            Positive scales for the data-derived upper bound of each free
            absorption flux. Multiple values enable internal multi-bound
            exploration.
        cancellation_threshold
            Cancellation fraction that triggers continuous refinement. Use
            ``None`` to disable the refinement layer.
        relative_likelihood_min
            Minimum likelihood relative to the global MLE for both discrete
            candidate selection and continuous refinement.
        random_seed
            Nonnegative seed used by random starts. Structured starts are
            deterministic and do not depend on this value.
        """
        from noobfriend.inference.spectrum.workspace.mle.fit import fit_workspace_mle

        return fit_workspace_mle(
            self,
            absorption_bound_multipliers=absorption_bound_multipliers,
            cancellation_threshold=cancellation_threshold,
            relative_likelihood_min=relative_likelihood_min,
            random_seed=random_seed,
        )

    def summary(self) -> str:
        """Return an HTML summary of this prepared fit plan."""
        from noobfriend.inference.spectrum.workspace.summary import (
            render_workspace_summary,
        )

        return render_workspace_summary(self)

    def __iter__(self):
        """Iterate over line handles."""
        return iter(self.handles)

    def __len__(self) -> int:
        """Return the number of prepared line handles."""
        return len(self.handles)


def _normalize_line_entries(
    lines: Sequence[NoobLine] | Mapping[str, NoobLine],
) -> tuple[Literal["auto", "manual"], tuple[_LineEntry, ...]]:
    if isinstance(lines, Mapping):
        entries: list[_LineEntry] = []
        for key, line in lines.items():
            if not isinstance(key, str) or not key:
                raise ValueError("manual line ids must be non-empty strings.")
            entries.append((key, line))
        if not entries:
            raise ValueError("prepare requires at least one line.")
        return "manual", tuple(entries)

    if isinstance(lines, (str, bytes)) or not isinstance(lines, Sequence):
        raise TypeError(
            "prepare expects a sequence of NoobLine or a mapping of id to NoobLine."
        )
    entries = tuple((None, line) for line in lines)
    if not entries:
        raise ValueError("prepare requires at least one line.")
    return "auto", entries


def _validate_line_objects(entries: tuple[_LineEntry, ...]) -> None:
    seen: dict[int, int] = {}
    for index, (_, line) in enumerate(entries):
        if not isinstance(line, NoobLine):
            raise TypeError("prepare entries must be NoobLine instances.")
        identity = id(line)
        if identity in seen:
            raise ValueError(
                "the same NoobLine object cannot appear twice in one workspace."
            )
        seen[identity] = index


def _validate_line_graph(entries: tuple[_LineEntry, ...]) -> None:
    identities = {id(line) for _, line in entries}
    for _, line in entries:
        if line.base is not None and id(line.base) not in identities:
            raise ValueError("derived line base is not in the workspace line list.")
        for rule in (line.center_rule, line.fwhm_rule, line.flux_rule):
            if rule.target is not None and id(rule.target) not in identities:
                raise ValueError(
                    "locked rule target is not in the workspace line list."
                )
        if line.flux_rule.is_ratio and (
            line.base is None or id(line.base) not in identities
        ):
            raise ValueError(
                "flux ratio requires its base line in the workspace line list."
            )
        _check_base_cycle(line)


def _check_base_cycle(line: NoobLine) -> None:
    seen: set[int] = set()
    current = line
    while current.base is not None:
        current_id = id(current)
        if current_id in seen:
            raise ValueError("line base graph contains a cycle.")
        seen.add(current_id)
        current = current.base


def _build_handles(
    spectrum: NoobSpectrum,
    entries: tuple[_LineEntry, ...],
    *,
    id_mode: Literal["auto", "manual"],
) -> tuple[LineHandle, ...]:
    resolved = [_resolve_line(spectrum, line) for _, line in entries]
    ids = (
        [manual_id for manual_id, _ in entries]
        if id_mode == "manual"
        else _auto_ids(resolved)
    )
    if len(set(ids)) != len(ids):
        raise ValueError("workspace line ids must be unique.")
    reserved = _RESERVED_LINE_IDS.intersection(ids)
    if reserved:
        names = ", ".join(repr(value) for value in sorted(reserved))
        raise ValueError(
            f"workspace line ids are reserved for model components: {names}."
        )
    return tuple(
        LineHandle(
            id=line_id,
            index=index,
            line=line,
            observed_wavelength=center,
            rest_wavelength=rest,
            component=line.component,
            contribution=line.contribution,
            profile=line.profile,
        )
        for index, (line_id, (line, center, rest)) in enumerate(
            zip(ids, resolved, strict=True)
        )
    )


def _resolve_line(
    spectrum: NoobSpectrum, line: NoobLine
) -> tuple[NoobLine, float, float | None]:
    if (
        spectrum.z is not None
        and line.z is not None
        and not isclose(spectrum.z, line.z, rel_tol=1e-10, abs_tol=1e-12)
    ):
        raise ValueError("line z is inconsistent with spectrum z.")

    center = convert_wavelength(
        line.observed_wavelength, from_unit=line.unit, to_unit=spectrum.unit
    )
    rest = (
        None
        if line.rest is None
        else convert_wavelength(line.rest, from_unit=line.unit, to_unit=spectrum.unit)
    )
    if spectrum.z is not None and line.z is None and rest is not None:
        expected = rest * (1.0 + spectrum.z)
        if not isclose(center, expected, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(
                "line rest/obs wavelengths are inconsistent with spectrum z."
            )
    if center < float(spectrum.obs[0]) or center > float(spectrum.obs[-1]):
        raise ValueError("line observed wavelength is outside the spectrum range.")
    return line, center, rest


def _auto_ids(resolved: list[tuple[NoobLine, float, float | None]]) -> list[str]:
    bases = [_base_label(line, center) for line, center, _ in resolved]
    base_counts = Counter(bases)

    component_candidates = [
        base if base_counts[base] == 1 else f"{base}.{line.component}"
        for base, (line, _, _) in zip(bases, resolved, strict=True)
    ]
    component_counts = Counter(component_candidates)

    profile_candidates = [
        candidate if component_counts[candidate] == 1 else f"{candidate}.{line.profile}"
        for candidate, (line, _, _) in zip(component_candidates, resolved, strict=True)
    ]
    profile_counts = Counter(profile_candidates)

    contribution_candidates = [
        candidate
        if profile_counts[candidate] == 1
        else f"{candidate}.{line.contribution}"
        for candidate, (line, _, _) in zip(profile_candidates, resolved, strict=True)
    ]
    contribution_counts = Counter(contribution_candidates)
    counters: defaultdict[str, int] = defaultdict(int)
    output: list[str] = []
    for candidate in contribution_candidates:
        if contribution_counts[candidate] == 1:
            output.append(candidate)
            continue
        counters[candidate] += 1
        output.append(f"{candidate}.{counters[candidate]}")
    return output


def _base_label(line: NoobLine, center: float) -> str:
    if line.linename:
        return line.linename
    return f"Line_obs_{_format_center_label(center)}"
