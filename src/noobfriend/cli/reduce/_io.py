"""Path and option resolution for the ``reduce`` command group."""

import os

from noobfriend.cli.reduce._recipe import Recipe, StepConfig, ordered
from noobfriend.cli.reduce._registry import Pipeline, for_recipe
from noobfriend.core.env import get_settings, stage_path_var


def _rendered_save_stages(recipe: Recipe, pipeline: Pipeline) -> list[str]:
    """Return the output stages the selected renderer will actually write."""
    stages = [
        cfg.save_as for _, cfg in ordered(pipeline.chain, recipe.steps) if cfg.save_as
    ]
    configured = {
        name: cfg.save_as for name, cfg in recipe.steps.items() if cfg.save_as
    }

    if pipeline.pupil == "grism" and pipeline.input_stage == "2a":
        # The two-pass GRISM renderer always materializes the pass-1 product;
        # the template pass is present unless template_bkg is explicitly skipped.
        stages.append(configured.get("flat", "2b"))
        template_cfg = recipe.steps.get("template_bkg", StepConfig())
        if not template_cfg.skip:
            stages.append(configured.get("template_bkg", "2bii"))
    elif pipeline.pupil == "clear" and pipeline.input_stage == "2bi":
        # Stage-3 resample is required and the renderer falls back to 3a when the
        # recipe omits an explicit save_as.
        stages.append(configured.get("resample", "3a"))

    return list(dict.fromkeys(stage for stage in stages if stage))


def validate_stages(recipe: Recipe) -> None:
    """Check the recipe produces output and that its stages are defined in the env.

    The input stage (``[select].stage``) and every stage the selected renderer
    writes must have a ``STAGE_<STAGE>_PATH`` set, since the generated script
    resolves the input and output directories from them.

    Raises
    ------
    ValueError
        If the renderer would produce no output stage, or any required stage has
        no ``STAGE_<STAGE>_PATH`` in the environment.
    """
    pipeline = for_recipe(recipe)
    save_stages = _rendered_save_stages(recipe, pipeline)
    if not save_stages:
        raise ValueError("recipe defines no output stage; nothing would be produced.")
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
