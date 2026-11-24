"""Picking a storage by URL protocol."""

from typing import Any

from fsspec.core import split_protocol  # type: ignore[import-untyped]  # fsspec ships no type hints

from policies.deploy.registry.storage.base import Storage, StorageConfig, storages
from policies.deploy.registry.storage.filesystem import FilesystemStorage


@storages.register("hf", StorageConfig)
def _hub(config: StorageConfig, **options: Any) -> Storage:
    # huggingface_hub is the optional `hub` extra, so it is imported on first use.
    from policies.deploy.registry.storage.hub import HubStorage

    return HubStorage.from_url(config.url, **options)


def open_storage(url: str, **options: Any) -> Storage:
    """The storage registered for the URL's protocol, else fsspec. `options` go to the backend."""
    protocol, _ = split_protocol(url)
    if protocol in storages:
        return storages.build({"type": protocol, "url": url}, **options)
    return FilesystemStorage.from_url(url, **options)
