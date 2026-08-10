"""Resolve the repo root across source checkouts and editable installs."""

import os


def repo_root(start: str | None = None) -> str:
    """
    Directory that contains ``pyproject.toml``.

    Walks up from ``start`` (default: this file) so strict editable installs
    under ``build/__editable__...`` still find the checkout root.
    """
    cur = os.path.dirname(os.path.abspath(start or __file__))
    while True:
        if os.path.isfile(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError(
        "Could not find repo root (no pyproject.toml above "
        f"{os.path.abspath(start or __file__)})"
    )


def data_cache_dir() -> str:
    """Repo ``01_data/cache`` (works under strict editable installs)."""
    return os.path.join(repo_root(), "01_data", "cache")
