"""SHA-256 of a value's canonical JSON."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

type Fingerprintable = (
    BaseModel | Mapping[str, Any] | Sequence[Any] | str | int | float | bool | None
)


def _plain(value: Fingerprintable) -> Any:
    # Models anywhere inside, in their canonical form.
    return to_jsonable_python(value, by_alias=True, round_trip=True)


def fingerprint(value: Fingerprintable) -> str:
    """Stable across key order and processes. NaN and infinity are rejected."""
    text = json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short(digest: str, length: int = 12) -> str:
    """Prefix for display, e.g. in artifact ids."""
    return digest[:length]
