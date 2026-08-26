"""Phase planning: an instruction becomes an ordered list of robot and human steps.

This is the layer that used to live inside a fork of the planner (``tiptop/hitl/``) and now lives
here, because it is not a planner's job. Given a workspace image and an instruction, a vision model

  1. breaks the instruction into an ORDERED list of phases, each one either a sub-goal for a task
     and motion planner or an action only a person can do;
  2. INVENTS a predicate where none of the planner's fit (``IsFolded(cloth)``), grounded by a
     natural-language description a vision model checks in an image;
  3. hands each human phase over as a teleop leg with written instructions;
  4. checks, from a fresh image, that the human's phase actually had the intended effect.

Each robot phase becomes one ordinary planner rollout aimed at that phase's sub-goal; each human
phase becomes one teleop leg. Legs of one task share a trajectory id, so they merge into a single
episode.

Nothing here knows which planner is behind it. What a planner can be asked for is declared in
``tandem.planners.base.Capabilities`` and read from there, so pointing tandem at a different task and
motion planner needs a new backend and changes nothing in this package.

Submodules are imported directly (``from tandem.planning import proposal``) rather than re-exported
here: this module must stay import-light, because ``tandem.planners.base`` imports ``symbols`` from
it and a re-export would close that loop.
"""
