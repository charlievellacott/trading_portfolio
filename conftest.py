from __future__ import annotations

def _require_editable_install() -> None:
    """
    The project uses numbered top-level directories (e.g. 01_data/), which cannot
    be imported directly in Python. Imports rely on the package-dir mapping in
    pyproject.toml, activated via an editable install.
    """

    try:
        import data  # noqa: F401
        import performance  # noqa: F401
        import risk  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Project packages are not importable.\n\n"
            "Run this once from the repo root:\n"
            "  python -m pip install -e .\n"
        ) from exc


_require_editable_install()

