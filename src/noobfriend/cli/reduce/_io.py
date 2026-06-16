"""Path and option resolution for the ``reduce`` command group."""

import os

from noobfriend.cli.reduce._recipe import Recipe, ordered
from noobfriend.cli.reduce._registry import for_recipe
from noobfriend.core.env import get_settings, stage_path_var


def validate_stages(recipe: Recipe) -> None:
    """Check the recipe produces output and that its stages are defined in the env.

    The input stage (``[select].stage``) and every ``save_as`` stage of the
    recipe's pipeline must have a ``STAGE_<STAGE>_PATH`` set, since the generated
    script resolves the input and output directories from them.

    Raises
    ------
    ValueError
        If no step has ``save_as`` (nothing would be produced), or any required
        stage has no ``STAGE_<STAGE>_PATH`` in the environment.
    """
    chain = for_recipe(recipe).chain
    save_stages = [
        cfg.save_as for _, cfg in ordered(chain, recipe.steps) if cfg.save_as
    ]
    if not save_stages:
        raise ValueError("recipe defines no save_as stage; nothing would be produced.")
    needed = list(dict.fromkeys([recipe.select.stage, *save_stages]))
    missing = [s for s in needed if os.getenv(stage_path_var(s)) is None]
    if missing:
        raise ValueError(
            f"stage(s) {missing} have no {stage_path_var('<STAGE>')} defined in the "
            "environment; add them (e.g. `noobfriend env add-stage`) before generating."
        )


def resolve_stage(stage: str | None) -> str:
    """Return the broad pipeline stage, falling back to ``START_STAGE``'s digit.

    Raises
    ------
    ValueError
        If ``stage`` is ``None`` and ``START_STAGE`` is unset.
    """
    if stage is not None:
        return stage
    configured = get_settings().start_stage
    if configured is None:
        raise ValueError("no stage given and START_STAGE is unset in the environment.")
    return configured[0]
