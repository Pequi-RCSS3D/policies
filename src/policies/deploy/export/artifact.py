"""Write a complete artifact directory: ONNX policy, weights and manifest."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from policies.deploy.export.canonical import canonical_spec
from policies.deploy.export.onnx import export_onnx
from policies.deploy.export.source import Exportable
from policies.deploy.inference.requirements import requirements
from policies.deploy.registry import Manifest, artifact_id, write_manifest


def export_artifact(
    source: Exportable,
    directory: str | Path,
    *,
    metrics: Mapping[str, float] | None = None,
    parents: Sequence[str] | None = None,
    run: Mapping[str, Any] | None = None,
) -> Manifest:
    """Ready to register: `policy.onnx`, `weights.pt` and `manifest.json`.

    `weights.pt` holds the actor's settings and state, enough to rebuild it exactly.
    `metrics` are scores to record, such as evaluation returns. `parents` are the ids of the
    artifacts this one derives from. They default to the source's `parents`, such as a
    `Learner`'s artifact teachers. `run` is free-form provenance.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    actor = source.actor
    spec = canonical_spec(actor.spec)
    # Once, so a component without a distribution warns once.
    required = requirements(spec)
    export_onnx(source, directory / "policy.onnx", required)
    torch.save({"settings": actor.settings, "actor": actor.state_dict()}, directory / "weights.pt")
    fingerprint = spec.fingerprint()
    manifest = Manifest(
        id=artifact_id(spec.name, fingerprint),
        name=spec.name,
        fingerprint=fingerprint,
        spec=spec.model_dump(mode="json"),
        requirements=required,
        metrics=dict(metrics or {}),
        parents=tuple(getattr(source, "parents", ()) if parents is None else parents),
        files=("policy.onnx", "weights.pt"),
        run=dict(run or {}),
    )
    write_manifest(directory, manifest)
    return manifest
