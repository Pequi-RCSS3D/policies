from collections.abc import Iterator

import pytest
import structlog


@pytest.fixture(autouse=True)
def reset_logging() -> Iterator[None]:
    """`main` configures structlog globally, so undo it after each test."""
    yield
    structlog.reset_defaults()
