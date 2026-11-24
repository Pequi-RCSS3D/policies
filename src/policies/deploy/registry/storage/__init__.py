"""Where artifacts live. Register a storage for a URL protocol with `storages.register`."""

from policies.deploy.registry.storage.base import (
    Storage,
    StorageConfig,
    StorageConflictError,
    storages,
)
from policies.deploy.registry.storage.filesystem import FilesystemStorage
from policies.deploy.registry.storage.open import open_storage

__all__ = [
    "FilesystemStorage",
    "Storage",
    "StorageConfig",
    "StorageConflictError",
    "open_storage",
    "storages",
]
