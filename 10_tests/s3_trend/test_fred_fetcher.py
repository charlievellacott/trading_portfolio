"""Offline unit tests for FRED fetcher (cache helpers / credential gate)."""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.ingestion.alternative_data.fred_fetcher import _cache_path, fetch_fred_series
from data.ingestion.credentials_env import read_credential, require_credential


def test_cache_path_helpers():
    p = _cache_path("DFF", cache_dir="/tmp/fred_test_cache")
    assert p.endswith("DFF.parquet")
    assert p == os.path.join("/tmp/fred_test_cache", "DFF.parquet")
    default = _cache_path("ECBDFR")
    assert default.endswith("ECBDFR.parquet")
    assert "fred" in default.replace("\\", "/")


def test_require_credential_raises_on_keyhere():
    path = os.path.join(ROOT, "config", "credentials.env")
    # Write a temp credentials file with placeholder and assert gate.
    with tempfile.TemporaryDirectory() as tmp:
        creds = os.path.join(tmp, "credentials.env")
        with open(creds, "w", encoding="utf-8") as f:
            f.write("FRED_API_KEY=keyhere\n")
        with pytest.raises(ValueError, match="keyhere"):
            require_credential("FRED_API_KEY", path=creds)
        with open(creds, "w", encoding="utf-8") as f:
            f.write("FRED_API_KEY=\n")
        with pytest.raises(ValueError):
            require_credential("FRED_API_KEY", path=creds)
    # Live repo file may also be keyhere — gate must still fire for that key.
    val = read_credential("FRED_API_KEY", path=path)
    if val == "keyhere" or not val:
        with pytest.raises(ValueError, match="FRED_API_KEY"):
            require_credential("FRED_API_KEY", path=path)


@pytest.mark.skipif(
    (read_credential("FRED_API_KEY") or "keyhere") == "keyhere",
    reason="FRED_API_KEY missing or placeholder keyhere",
)
def test_fred_fetch_optional_when_key_present():
    s = fetch_fred_series("DFF", start="2024-01-02", end="2024-01-10")
    assert s is not None
    assert len(s) >= 1
    assert s.name == "DFF"
