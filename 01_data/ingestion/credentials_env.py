"""Parse ``config/credentials.env`` (KEY=value lines; no python-dotenv)."""

from __future__ import annotations

import os

from data.repo_paths import repo_root


def credentials_env_path() -> str:
    return os.path.join(repo_root(), "config", "credentials.env")


def read_credential(key: str, path: str | None = None) -> str | None:
    """Return the value for ``key`` or ``None`` if missing / empty."""
    creds_path = credentials_env_path() if path is None else path
    if not os.path.isfile(creds_path):
        return None
    prefix = f"{key}="
    with open(creds_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if ln.startswith(prefix):
                val = ln.split("=", 1)[1].strip()
                return val or None
    return None


def require_credential(key: str, path: str | None = None) -> str:
    val = read_credential(key, path=path)
    if not val or val == "keyhere":
        creds_path = credentials_env_path() if path is None else path
        raise ValueError(
            f"Set a real {key}=... in {creds_path} (placeholder 'keyhere' is not valid)."
        )
    return val


def ensure_credential_placeholders(
    keys: dict[str, str],
    path: str | None = None,
) -> list[str]:
    """Append missing ``KEY=default`` lines. Does not overwrite existing keys.

    Returns the list of keys that were appended.
    """
    creds_path = credentials_env_path() if path is None else path
    existing: set[str] = set()
    lines: list[str] = []
    if os.path.isfile(creds_path):
        with open(creds_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        for ln in lines:
            s = ln.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            existing.add(s.split("=", 1)[0].strip())
    appended: list[str] = []
    for key, default in keys.items():
        if key in existing:
            continue
        lines.append(f"{key}={default}")
        appended.append(key)
    if appended:
        os.makedirs(os.path.dirname(creds_path) or ".", exist_ok=True)
        with open(creds_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")
    return appended
