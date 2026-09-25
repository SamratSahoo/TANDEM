"""The TiPToP backend: a capability declaration, a factory, a parent-side client, and a sidecar.

``capabilities.CAPABILITIES`` is what tandem's phase planner reads. ``FACTORY`` is what the registry
holds under the name ``tiptop`` -- it describes TiPToP to a catalog, installs its runtime, and builds
the ``backend.TiptopBackend`` a session drives. ``sidecar.py`` is the half that runs inside the GPU
runtime -- see its docstring for why it is a script rather than a module.

Nothing here is imported unless a profile actually asks for this backend, and nothing here imports
torch, cuTAMP or tiptop in the parent process.
"""

from tandem.planners.tiptop.capabilities import CAPABILITIES
from tandem.planners.tiptop.factory import FACTORY

__all__ = ["CAPABILITIES", "FACTORY"]
