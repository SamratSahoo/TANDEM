"""Who carries out a human phase: pi_omega_Delta in the paper, chosen by ``hitl.human_executor``.

``base`` holds the protocol, what a leg is asked for and answers with, and the registry. ``teleop``
holds the executor that ships: a person driving the arm through the DROID teleop driver. Another
package adds one through the ``tandem.human_executors`` entry point, or at runtime through
``register_human_executor``. See ``base`` for what an executor has to do.

Importing this package imports neither ``teleop`` nor any plugin, so listing executors works on a
laptop with no DROID install.
"""

from tandem.executors.base import (
    ENTRY_POINT_GROUP,
    SEGMENT_SOURCES,
    CustodyError,
    ExecutorContext,
    ExecutorFactory,
    ExecutorInfo,
    HumanExecutor,
    HumanPhaseRequest,
    HumanPhaseResult,
    available,
    catalog,
    check_name,
    create,
    info,
    refresh,
    register_human_executor,
    unregister_human_executor,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "SEGMENT_SOURCES",
    "CustodyError",
    "ExecutorContext",
    "ExecutorFactory",
    "ExecutorInfo",
    "HumanExecutor",
    "HumanPhaseRequest",
    "HumanPhaseResult",
    "available",
    "catalog",
    "check_name",
    "create",
    "info",
    "refresh",
    "register_human_executor",
    "unregister_human_executor",
]
