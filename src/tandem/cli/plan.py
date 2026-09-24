"""`tandem plan` — decompose a task from a photo, with no robot and no GPU.

The cheapest way to find out whether an instruction decomposes the way you meant, and the one to
reach for when a session produces a plan that looks wrong. It prints the ordered phases, who does
each, the sub-goal handed to the planner, each human phase's operator (what it needs, makes true
and undoes), whether those contracts hang together across the plan, the invented predicates and
their classifiers, and anything the proposer could not express.

This command is only possible because the phase planner is tandem's own now. It used to live inside
the planner's process, so answering "is this decomposition right?" meant a warm cuRobo, an open
camera and an arm — and the answer arrived with an operator already standing next to it. The remedy
for a plan that covers less than it was asked (put the missing object on the table, rewrite the
instruction) is only available *before* the arm moves, so being able to ask beforehand is the point.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from tandem.cli import theme
from tandem.core.errors import TandemError
from tandem.planners import registry


def plan(
    goal: str = typer.Argument(..., help="The instruction to decompose."),
    image: Path = typer.Option(
        ..., "--image", "-i", exists=True, dir_okay=False, help="A photo of the workspace."
    ),
    objects: list[str] = typer.Option(
        None,
        "--object",
        "-o",
        help="Pin an object label instead of asking a model to name them. Repeatable. Use it to "
        "reproduce a session's decomposition from the labels its perception actually produced.",
    ),
    backend: str = typer.Option(
        None,
        "--backend",
        "-b",
        help="Whose goal language to plan in: by default --profile's planner, else the machine's default "
        f"planner. One of: {', '.join(registry.available())}.",
    ),
    profile_name: str = typer.Option(
        None, "--profile", "-p", help="Take the planning settings from this profile."
    ),
    table: str = typer.Option("table", "--table", help="What the planner calls the table surface."),
    as_json: bool = typer.Option(False, "--json", help="Print the plan as JSON instead of prose."),
    save_vlm_io: Path = typer.Option(
        None, "--save-vlm-io", help="Write every image sent to the model, and its reply, here."
    ),
) -> None:
    from tandem.planning import objects as objects_mod
    from tandem.planning.grounding import descriptions_for, to_pil
    from tandem.planning.plan import build_plan
    from tandem.planning.record import recording_to
    from tandem.planning.symbols import ProposalError, describe

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a base dependency
        raise TandemError("Pillow is needed to read the workspace photo.", hint="pip install pillow") from exc

    backend = backend or _planner_for(profile_name)
    caps = registry.capabilities(backend)
    cfg = _config_for(profile_name)
    picture = to_pil(Image.open(image).convert("RGB"))

    async def run():
        names = objects_mod.sanitize_all(objects or [])
        if not names:
            _status(as_json, theme.busy, "naming the objects in the photo")
            names = await objects_mod.detect_objects(picture, goal, cfg)
        _status(as_json, theme.info, "objects", ", ".join(names))
        return names, await build_plan(picture, goal, names, table, cfg, caps, trajectory_id=None)

    with recording_to(save_vlm_io):
        try:
            names, (built, failure) = asyncio.run(run())
        except ProposalError as exc:
            raise TandemError(
                "The model could not produce a usable plan for that instruction.", hint=str(exc)
            ) from exc

    if built is None:
        raise TandemError(failure or "the plan could not be built", hint="Try rewording the instruction.")

    if as_json:
        # Written straight to stdout, not through the console: rich would wrap and colour it, and
        # anything else printed on the way here would have to be on stderr for the result to parse.
        # `tandem plan --json | jq` is the whole point of the flag.
        typer.echo(json.dumps(built.to_json(), indent=2))
        return

    spec = built.spec
    descriptions = descriptions_for(spec.invented, caps)
    theme.blank()
    theme.heading(spec.instruction, f"{len(spec.phases)} phase(s) · planner: {caps.name}")

    for i, phase in enumerate(spec.phases):
        who = "human" if phase.is_human else "robot"
        style = "violet" if phase.is_human else "accent"
        theme.console().print(f"  [{style}]{i}  {who:<5}[/{style}]  {phase.description}")
        for atom in sorted(phase.atoms, key=str):
            theme.console().print(f"        [faint]{theme.DOT}[/faint] {describe(atom, descriptions)}")
        if phase.is_human:
            theme.console().print(f"        [faint]{phase.instructions}[/faint]")
            if phase.operator is not None:
                _print_operator(phase.operator)
        else:
            rendered = [a.to_dict() for a in _goal_of(phase, caps)]
            theme.console().print(f"        [faint]goal: {json.dumps(rendered)}[/faint]")

    theme.blank()
    _print_contract_check(built, cfg, caps)

    if spec.invented:
        theme.blank()
        theme.rule("invented predicates")
        for predicate in spec.invented:
            arity = ", ".join(p.type for p in predicate.predicate.parameters)
            theme.console().print(f"  [violet]{predicate.name}[/violet]({arity})")
            theme.console().print(f"      [faint]{predicate.instructions}[/faint]")

    if spec.unrepresented:
        theme.blank()
        theme.error_panel(
            "\n".join(f"{u['clause']}\n    {u['reason']}" for u in spec.unrepresented),
            "This run would do LESS than it was asked. Put the missing object on the table, or "
            "reword the instruction, before collecting.",
        )
    elif not spec.needs_human:
        theme.blank()
        theme.info("no human phases", "the planner can do this whole task on its own")


def _print_operator(operator) -> None:
    """A human phase's magic operator: the lifted signature, then its whole contract.

    All three lists are printed, empty ones included. "Deletes nothing" is a statement the model made
    about the step, and the one most often got wrong -- a missing line would hide exactly that.
    """
    from rich.markup import escape

    def atoms(found) -> str:
        return escape(", ".join(sorted(str(a) for a in found))) or "none"

    console = theme.console()
    console.print(
        f"        [violet]operator[/violet] {escape(operator.signature)}  "
        f"[faint]as {escape(operator.display)}[/faint]"
    )
    console.print(f"          [faint]preconditions [/faint] {atoms(operator.preconditions)}")
    console.print(f"          [faint]add effects   [/faint] {atoms(operator.add_effects)}")
    console.print(f"          [faint]delete effects[/faint] {atoms(operator.delete_effects)}")


def _print_contract_check(built, cfg, caps) -> None:
    """Whether the phases hang together as a plan, as the proposal stage judged it.

    An accepted plan has already passed ``check_plan_effects`` inside the repair loop when the
    profile has it on, so this re-runs it -- it is set arithmetic, no model call -- to say so rather
    than leave the reader to infer it. With it off, the same check is still reported, because
    "would this have been refused?" is the question this command exists to answer before the arm
    moves. A repeated robot move is only ever a warning (see ``contracts.wasted_robot_move``).

    With ``classify_initial`` on, the plan was then held to the starting state the photo was
    measured to show (``PhasePlan.recheck_plan_effects``). That second check can prove what the
    first could not -- a precondition no phase establishes and the scene does not already satisfy
    -- and a session only records it, since the proposer is out of the loop by then. It is printed
    here because this is the one place it can still be acted on: reword, or set the scene up.
    """
    from tandem.planning import contracts

    spec = built.spec
    broken = contracts.check_plan_effects(spec, caps=caps)
    if broken is None:
        detail = "checked in the repair loop" if cfg.check_plan_effects else "check_plan_effects is off"
        theme.ok("the phases' contracts hang together", detail)
    elif cfg.check_plan_effects:  # pragma: no cover - the repair loop refuses such a plan
        theme.fail("the phases' contracts do not hang together", broken)
    else:
        theme.warn("the phases' contracts do not hang together (check_plan_effects is off)", broken)
    if built.plan_effects_rechecked:
        if built.inconsistency:
            theme.warn("the plan does not hang together against the scene in the photo", built.inconsistency)
        else:
            theme.ok("the plan holds against the scene in the photo", "the starting state was measured")
    wasted = contracts.wasted_robot_move(spec.phases, caps=caps)
    if wasted:
        theme.warn("this plan repeats work", wasted)


def _status(as_json: bool, printer, *args) -> None:
    """Progress chatter, silenced under --json so the payload is the only thing on stdout."""
    if not as_json:
        printer(*args)


def _goal_of(phase, caps):
    from tandem.planners.base import to_goal_atoms

    return to_goal_atoms(sorted(phase.atoms, key=str), caps)


def _planner_for(profile_name: str | None) -> str:
    """The planner a plan is proposed for when --backend does not say: the profile's, or the machine's default."""
    from tandem.core import profiles
    from tandem.core import settings as settings_mod

    if profile_name:
        return profiles.load(profile_name).planner.backend
    return settings_mod.load().default_planner


def _config_for(profile_name: str | None):
    """Planning settings from a profile, or the defaults with planning turned on.

    ``enabled`` is forced on: asking for a plan IS asking for one, and refusing because a profile has
    the feature switched off for collection would be obtuse.
    """
    import dataclasses

    from tandem.planning.config import PlanningConfig

    if not profile_name:
        return PlanningConfig(enabled=True)

    from tandem.core import profiles

    profile = profiles.load(profile_name)
    # Through the profile, so this reads the same file a collection session would rather than one
    # relative to wherever the command was run.
    cache = profiles.resolve_cache_path(profile)
    return dataclasses.replace(profile.hitl.to_planning_config(cache_path=cache), enabled=True)
