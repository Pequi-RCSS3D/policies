"""An exported policy as a teacher, batched over environments."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from policies.core.logs import get_logger
from policies.deploy.inference import Policy, PolicyLoadError
from policies.deploy.registry import artifact_id
from policies.io.bookkeeping import Compose
from policies.learning.teachers.reference import TeacherSource, artifact_path

log = get_logger(__name__)


class ArtifactTeacher:
    """Runs an artifact with ONNX Runtime, one call for all environments.

    Keeps its own bookkeeping and recurrent state, so it may differ from the student's.
    """

    def __init__(self, path: str | Path, threads: int = 1) -> None:
        self.policy = Policy.load(path, threads)
        fixed = [
            value.name
            for value in self.policy.session.get_inputs()
            if isinstance(value.shape[0], int)
        ]
        if fixed:
            raise PolicyLoadError(f"{path}: fixed batch size in {fixed}. Re-export it.")
        self.sources = set(self.policy.observation_keys)
        """Observation keys the teacher reads. Others are not copied to the host."""
        self.artifact_id = artifact_id(self.policy.spec.name, self.policy.fingerprint)
        """The id the artifact registers under, among a distilled student's `parents`."""
        self.bookkeeping: Compose | None = None
        self.batch = 0
        self.state: dict[str, np.ndarray] = {}

    @classmethod
    def load(cls, source: TeacherSource, threads: int = 1) -> "ArtifactTeacher":
        path = artifact_path(source)
        log.info("teacher loaded", path=str(path))
        return cls(path, threads)

    def _start(self, batch: int) -> None:
        self.batch = batch
        self.bookkeeping = self.policy.initial_bookkeeping((batch,))
        self.state = self.policy.initial_state(batch)

    def __call__(self, observation: Mapping[str, Tensor], reset: Tensor) -> dict[str, Tensor]:
        """Actions for every environment: the mode of each part's distribution."""
        restart = reset.detach().cpu().numpy().astype(bool)
        if self.bookkeeping is None or len(restart) != self.batch:
            self._start(len(restart))
        else:
            self.bookkeeping.reset(restart)
            keep = ~restart
            self.state = {
                name: value * keep.reshape(-1, *[1] * (value.ndim - 1))
                for name, value in self.state.items()
            }
        assert self.bookkeeping is not None
        arrays = {
            name: value.detach().cpu().numpy()
            for name, value in observation.items()
            if name in self.sources
        }
        parts = self.policy.inputs(arrays, self.bookkeeping.observe(arrays))
        raw, self.state = self.policy.run(parts, self.state)
        actions = self.policy.decoder.mode(raw)
        self.bookkeeping.commit(actions)
        return {name: torch.from_numpy(value).to(reset.device) for name, value in actions.items()}
