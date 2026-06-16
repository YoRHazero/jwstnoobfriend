"""Registry mapping ``(pupil, stage)`` to its reduction chain, scaffold and renderer.

This is the single place that knows the available pipelines. ``init`` and ``gen``
look pipelines up here (or iterate them all), and :class:`~noobfriend.cli.reduce._recipe.Recipe`
validation borrows :func:`step_names_for`. Adding a pipeline (e.g. a stage-3 one)
is a new ``_reduce_*`` / ``_codegen_*`` module pair plus one entry below.
"""

from collections.abc import Callable
from dataclasses import dataclass

from noobfriend.cli.reduce import (
    _codegen_clear_stage2,
    _codegen_grism_stage2,
    _reduce_clear_stage2,
    _reduce_grism_stage2,
)
from noobfriend.cli.reduce._recipe import Recipe, StepSpec


@dataclass(frozen=True)
class Pipeline:
    """One registered reduction pipeline, keyed by ``(pupil, stage)``."""

    pupil: str
    stage: str
    input_stage: str
    chain: tuple[StepSpec, ...]
    scaffold: Callable[[str], str]
    render: Callable[[Recipe], str]

    @property
    def recipe_name(self) -> str:
        """Conventional recipe filename, ``recipe_<pupil>_stage<stage>.toml``."""
        return f"recipe_{self.pupil}_stage{self.stage}.toml"

    @property
    def script_name(self) -> str:
        """Conventional runner filename, ``reduce_<pupil>_stage<stage>.py``."""
        return f"reduce_{self.pupil}_stage{self.stage}.py"


PIPELINES: dict[tuple[str, str], Pipeline] = {
    ("clear", "2"): Pipeline(
        "clear",
        "2",
        _reduce_clear_stage2.INPUT_STAGE,
        _reduce_clear_stage2.CHAIN,
        _reduce_clear_stage2.scaffold,
        _codegen_clear_stage2.render,
    ),
    ("grism", "2"): Pipeline(
        "grism",
        "2",
        _reduce_grism_stage2.INPUT_STAGE,
        _reduce_grism_stage2.CHAIN,
        _reduce_grism_stage2.scaffold,
        _codegen_grism_stage2.render,
    ),
}


def _broad_stage(select_stage: str) -> str:
    """Return the broad stage digit (``"2a"`` -> ``"2"``) used as the registry key."""
    return select_stage[0] if select_stage else ""


def lookup(pupil: str, stage: str) -> Pipeline:
    """Return the pipeline for ``(pupil, stage)``.

    Raises
    ------
    KeyError
        If no pipeline is registered for the pair.
    """
    try:
        return PIPELINES[(pupil, stage)]
    except KeyError:
        valid = ", ".join(f"{p}/stage{s}" for p, s in PIPELINES)
        raise KeyError(
            f"no pipeline for pupil={pupil!r} stage={stage!r}; registered: {valid}."
        ) from None


def for_recipe(recipe: Recipe) -> Pipeline:
    """Return the pipeline a recipe targets (from its ``pipeline`` and select stage)."""
    return lookup(recipe.pipeline, _broad_stage(recipe.select.stage))


def all_pipelines() -> list[Pipeline]:
    """Return every registered pipeline."""
    return list(PIPELINES.values())


def select_pipelines(pupil: str | None, stage: str | None) -> list[Pipeline]:
    """Return the registered pipelines matching ``pupil`` and/or ``stage``.

    A ``None`` filter matches everything, so passing neither returns all
    pipelines (the "generate everything" default of ``init`` / ``gen``).
    """
    return [
        pipeline
        for pipeline in PIPELINES.values()
        if (pupil is None or pipeline.pupil == pupil)
        and (stage is None or pipeline.stage == stage)
    ]


def step_names_for(pupil: str, select_stage: str) -> frozenset[str]:
    """Return the valid step names for a pipeline, or empty if it is unknown."""
    pipeline = PIPELINES.get((pupil, _broad_stage(select_stage)))
    return frozenset(spec.name for spec in pipeline.chain) if pipeline else frozenset()
