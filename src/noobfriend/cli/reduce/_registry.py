"""Registry mapping ``(pupil, input_stage)`` to its reduction chain and renderer.

This is the single place that knows the available pipelines. ``init`` and ``gen``
look pipelines up here (or iterate them all), and :class:`~noobfriend.cli.reduce._recipe.Recipe`
validation borrows :func:`step_names_for`. Adding a pipeline (e.g. a stage-3 one)
is a new ``_reduce_*`` / ``_codegen_*`` module pair plus one entry below.

Pipelines are keyed by ``(pupil, input_stage)`` -- the *full* input stage, not
its leading digit -- so two pipelines of the same flavour that consume the same
broad level stay distinct (CLEAR stage-2 reduces ``2a``; CLEAR stage-3 mosaics
``2bi``; both are level 2). ``Pipeline.stage`` carries the broad *output* digit
used for recipe / script filenames and CLI ``--stage`` filtering.
"""

from collections.abc import Callable
from dataclasses import dataclass

from noobfriend.cli.reduce import (
    _codegen_clear_stage2,
    _codegen_clear_stage3,
    _codegen_grism_stage2,
    _reduce_clear_stage2,
    _reduce_clear_stage3,
    _reduce_grism_stage2,
)
from noobfriend.cli.reduce._recipe import Recipe, StepSpec


@dataclass(frozen=True)
class Pipeline:
    """One registered reduction pipeline, keyed by ``(pupil, input_stage)``."""

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
    ("clear", "2a"): Pipeline(
        "clear",
        "2",
        _reduce_clear_stage2.INPUT_STAGE,
        _reduce_clear_stage2.CHAIN,
        _reduce_clear_stage2.scaffold,
        _codegen_clear_stage2.render,
    ),
    ("clear", "2bi"): Pipeline(
        "clear",
        "3",
        _reduce_clear_stage3.INPUT_STAGE,
        _reduce_clear_stage3.CHAIN,
        _reduce_clear_stage3.scaffold,
        _codegen_clear_stage3.render,
    ),
    ("grism", "2a"): Pipeline(
        "grism",
        "2",
        _reduce_grism_stage2.INPUT_STAGE,
        _reduce_grism_stage2.CHAIN,
        _reduce_grism_stage2.scaffold,
        _codegen_grism_stage2.render,
    ),
}


def lookup(pupil: str, input_stage: str) -> Pipeline:
    """Return the pipeline for ``(pupil, input_stage)``.

    Raises
    ------
    KeyError
        If no pipeline is registered for the pair.
    """
    try:
        return PIPELINES[(pupil, input_stage)]
    except KeyError:
        valid = ", ".join(f"{p}/{s}" for p, s in PIPELINES)
        raise KeyError(
            f"no pipeline for pupil={pupil!r} input_stage={input_stage!r}; "
            f"registered: {valid}."
        ) from None


def for_recipe(recipe: Recipe) -> Pipeline:
    """Return the pipeline a recipe targets (from its ``pipeline`` and select stage)."""
    return lookup(recipe.pipeline, recipe.select.stage)


def all_pipelines() -> list[Pipeline]:
    """Return every registered pipeline."""
    return list(PIPELINES.values())


def select_pipelines(pupil: str | None, stage: str | None) -> list[Pipeline]:
    """Return the registered pipelines matching ``pupil`` and/or output ``stage``.

    ``stage`` is the broad *output* digit (``"2"`` / ``"3"``), matched against
    :attr:`Pipeline.stage`. A ``None`` filter matches everything, so passing
    neither returns all pipelines (the "generate everything" default of ``init``
    / ``gen``).
    """
    return [
        pipeline
        for pipeline in PIPELINES.values()
        if (pupil is None or pipeline.pupil == pupil)
        and (stage is None or pipeline.stage == stage)
    ]


def step_names_for(pupil: str, input_stage: str) -> frozenset[str]:
    """Return the valid step names for a pipeline, or empty if it is unknown."""
    pipeline = PIPELINES.get((pupil, input_stage))
    return frozenset(spec.name for spec in pipeline.chain) if pipeline else frozenset()
