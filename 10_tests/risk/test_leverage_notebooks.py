"""Execute both 01_leverage notebooks on synthetic unlevered base returns."""

from __future__ import annotations

import os

import nbformat
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

S1_NB = os.path.join(ROOT, "06_risk", "notebooks", "s1_equities", "01_leverage.ipynb")
S2_NB = os.path.join(ROOT, "06_risk", "notebooks", "s2_coint", "01_leverage.ipynb")


def _synthetic_base(n: int, *, freq: str, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    if freq == "W":
        idx = pd.bdate_range("2016-01-04", periods=n, freq="W-MON")
        r = rng.normal(0.002, 0.015, n)
    else:
        idx = pd.bdate_range("2018-01-02", periods=n, freq="B")
        r = rng.normal(0.0004, 0.008, n)
    s = pd.Series(r, index=idx, name="ret")
    return s


def _assign_base_cell(series: pd.Series, parquet_path: str) -> str:
    series.to_frame().to_parquet(parquet_path)
    return (
        "import pandas as pd\n"
        f"BASE = pd.read_parquet(r{parquet_path!r})['ret']\n"
        "BASE.index = pd.to_datetime(BASE.index)\n"
        "print('synthetic BASE', len(BASE), BASE.index.min().date(), BASE.index.max().date())\n"
    )


def _prepare_notebook(
    path: str,
    series: pd.Series,
    parquet_path: str,
    artifact_path: str,
) -> nbformat.NotebookNode:
    nb = nbformat.read(path, as_version=4)
    cfg = "".join(nb.cells[2].source)
    cfg = cfg.replace(
        "DEFAULT_TARGETS = [0.06, 0.08, 0.10, 0.12, 0.15, 0.18]",
        "DEFAULT_TARGETS = [0.08, 0.12]",
    )
    cfg = cfg.replace("DEFAULT_DD_CAP = 0.25", "DEFAULT_DD_CAP = 0.80")
    # keep artifact_path assignment but override after
    nb.cells[2].source = cfg + f"\nARTIFACT_PATH = r{artifact_path!r}\n"
    nb.cells[4].source = _assign_base_cell(series, parquet_path)
    return nb


def _execute(nb: nbformat.NotebookNode) -> None:
    pytest.importorskip("nbclient")
    from jupyter_client.kernelspec import NoSuchKernel
    from nbclient import NotebookClient

    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": ROOT}},
    )
    try:
        client.execute()
    except NoSuchKernel:
        pytest.skip("python3 jupyter kernel is not installed")


def _assert_opening_notes(nb: nbformat.NotebookNode) -> None:
    text = "".join(nb.cells[0].source)
    assert "Run first" in text
    assert "Half-Kelly is betting about half" in text
    assert "CAGR is the constant yearly rate" in text
    assert "Calmar is that CAGR" in text
    assert "CVaR is the average outcome" in text


def test_s1_leverage_notebook_executes(tmp_path):
    series = _synthetic_base(80, freq="W", seed=7)
    nb = _prepare_notebook(
        S1_NB,
        series,
        os.path.join(str(tmp_path), "s1_base.parquet"),
        os.path.join(str(tmp_path), "s1_leverage.json"),
    )
    _assert_opening_notes(nb)
    _execute(nb)
    sources = ["".join(c.source) for c in nb.cells]
    assert any("half-Kelly" in s or "Half-Kelly" in s for s in sources)


def test_s2_leverage_notebook_executes(tmp_path):
    series = _synthetic_base(90, freq="D", seed=8)
    nb = _prepare_notebook(
        S2_NB,
        series,
        os.path.join(str(tmp_path), "s2_base.parquet"),
        os.path.join(str(tmp_path), "s2_leverage.json"),
    )
    _assert_opening_notes(nb)
    _execute(nb)
    sources = ["".join(c.source) for c in nb.cells]
    assert any("s1_vt" in s for s in sources)
