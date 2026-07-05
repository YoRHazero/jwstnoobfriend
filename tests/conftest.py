"""Shared test fixtures: keep the ambient environment out of the test run.

``_parse_spec`` resolves ``host:path`` specs through the mount/own-machine
resolver (:func:`noobfriend.core.env.to_local`), which consults ``NOOB_SERVER``
and ``DATA_ROOT_PATH``. A developer shell that exports those (e.g. on the data
server) would silently flip remote specs to local in unrelated tests, so they
are cleared here; tests that need them set them explicitly.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_storage_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the storage-location variables from the inherited environment."""
    monkeypatch.delenv("NOOB_SERVER", raising=False)
    monkeypatch.delenv("DATA_ROOT_PATH", raising=False)
