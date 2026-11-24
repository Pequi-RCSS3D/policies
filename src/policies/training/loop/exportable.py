"""Agents the trainer can export."""

from typing import Protocol, runtime_checkable

from policies.deploy.export import Exportable


@runtime_checkable
class ExportableAgent(Exportable, Protocol):
    """Any agent with what `export_artifact` needs. `Learner` is one."""
