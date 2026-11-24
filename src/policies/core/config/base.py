"""Base config model and YAML round-trip."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """Immutable config. Unknown keys are errors. Defaults are validated like given values."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


def canonical(config: BaseModel) -> dict[str, Any]:
    """JSON-compatible form, with every default filled in, that validates back to `config`."""
    return config.model_dump(mode="json", by_alias=True, round_trip=True)


def load_yaml[C: BaseModel](path: str | Path, model: type[C]) -> C:
    """Read and validate a YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return model.model_validate(data or {})


def dump_yaml(config: BaseModel, path: str | Path) -> None:
    """Write a config in canonical form."""
    text = yaml.safe_dump(canonical(config), sort_keys=False, allow_unicode=True)
    Path(path).write_text(text, encoding="utf-8")
