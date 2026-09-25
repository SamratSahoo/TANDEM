"""Scaffolding shared by more than one test module.

It lives here rather than in whichever test module happened to define it first. A test
module that imports another has to name it — `from tests.test_session import ...` — and that
only resolves when the repository root is on `sys.path`, which `python -m pytest` arranges
and a bare `pytest` does not. The suite passed locally and failed in CI for exactly that
reason.

pytest puts this directory on `sys.path` for every test module it collects, so `from helpers
import ...` works under either invocation. It also stops collecting one module from importing
another's fixtures and module-level state as a side effect.
"""

from __future__ import annotations

import time


class FakeFactory:
    """A planner factory that builds stand-in backends, and keeps the context each was built from.

    It implements ``tandem.planners.base.BackendFactory`` in full, so registering it exercises the
    same checks a real plugin goes through.
    """

    def __init__(self, name: str = "tiptop", *, backend_type=None, **backend_kwargs) -> None:
        from fake_backend import FakeBackend

        from tandem.planners.base import PlannerInfo

        self.info = PlannerInfo(name=name, display_name=f"fake {name}", summary="A planner with no planner behind it.")
        self.backend_type = backend_type or FakeBackend
        self.backend_kwargs = backend_kwargs
        self.contexts: list = []
        self.built: list = []

    def capabilities(self):
        from tandem.planners.tiptop.capabilities import CAPABILITIES

        return CAPABILITIES

    def create(self, ctx):
        self.contexts.append(ctx)
        backend = self.backend_type(
            None,
            output_dir=ctx.output_dir,
            execute=ctx.execute,
            record=ctx.record,
            on_log=ctx.on_log,
            **self.backend_kwargs,
        )
        self.built.append(backend)
        return backend

    def runtime(self, settings=None):
        return None


def isolate_registry(monkeypatch) -> None:
    """Undo, at teardown, everything a test registers with the planner registry."""
    from tandem.planners import registry

    monkeypatch.setattr(registry, "_registered", dict(registry._registered))
    monkeypatch.setattr(registry, "_loaded", dict(registry._loaded))


def use_fake_backend(monkeypatch, *, backend_type=None, **kwargs):
    """Make every session in this test build a FakeBackend, and return the list of those it builds.

    Registered in the planner registry under the name the profile already uses, so `planner.backend`
    stays a real name and the session takes exactly the path it takes in production: the registry,
    a factory, a BackendContext.
    """
    from tandem.planners import registry

    isolate_registry(monkeypatch)
    factory = FakeFactory("tiptop", backend_type=backend_type, **kwargs)
    registry.register_backend("tiptop", factory, replace=True)
    return factory.built


class FakeGemini:
    """A model client that answers by the KIND of question it was asked.

    Keying on the prompt rather than on call order is what makes it usable: a test that cares about
    verification should not have to know how many times the engine will re-propose a plan, and one
    that cares about re-planning should not have to pad a list with verdicts. Getting that wrong
    does not fail cleanly either — a verdict handed to the proposal parser is rejected as "the plan
    must contain at least one phase" and reprompted three times before the run gives up.
    """

    #: Text unique to each prompt, from tandem/planning/prompts.py.
    PLAN_MARKER = "ORDERED list of phases"
    CLASSIFY_MARKER = "Statement:"

    def __init__(self, plan: str, verdicts=None):
        self.plan = plan
        # The last verdict repeats, so "it never verifies" needs one entry rather than a guess at
        # how many retries the config allows.
        self.verdicts = list(verdicts or [])
        self.prompts: list[str] = []
        self.plan_calls = 0
        self.verdict_calls = 0
        self.aio = self

    @property
    def models(self):
        return self

    async def generate_content(self, model, contents, config):
        from unittest import mock

        prompt = contents[-1]
        self.prompts.append(prompt)
        if self.CLASSIFY_MARKER in prompt and self.PLAN_MARKER not in prompt:
            self.verdict_calls += 1
            if not self.verdicts:
                raise AssertionError("the model was asked to classify but no verdict was provided")
            text = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
        else:
            self.plan_calls += 1
            text = self.plan
        return mock.Mock(text=text)


def wait_for(predicate, timeout: float = 8.0, interval: float = 0.02) -> bool:
    """Poll until a predicate holds. Returns False on timeout so the caller can assert.

    The session engine is driven by threads reading pipes and a file tailer, so state changes land
    asynchronously; sleeping a fixed amount instead would be both slower and flakier.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def builtin_path(name: str):
    """The packaged copy of one of the paper's five (``profiles.BUILTIN``), as a file to read in a test."""
    from tandem import resources

    return resources.path(f"profiles/{name}.yml")
