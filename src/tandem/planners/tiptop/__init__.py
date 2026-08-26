"""The TiPToP backend: a capability declaration, a parent-side client, and a sidecar.

``capabilities.CAPABILITIES`` is what tandem's phase planner reads. ``backend.TiptopBackend`` is what
a session drives. ``sidecar.py`` is the half that runs inside the GPU runtime -- see its docstring
for why it is a script rather than a module.

Nothing here is imported unless a profile actually asks for this backend, and nothing here imports
torch, cuTAMP or tiptop in the parent process.
"""

from tandem.planners.tiptop.capabilities import CAPABILITIES

__all__ = ["CAPABILITIES"]
