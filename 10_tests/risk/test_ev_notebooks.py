"""Execute both EV vs SPY notebooks on synthetic sealed returns (no vendor fetch)."""

from __future__ import annotations

import os

import nbformat
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

S1_NB = os.path.join(ROOT, "06_risk", "notebooks", "s1_equities", "02_ev_vs_spy.ipynb")
S2_NB = os.path.join(ROOT, "06_risk", "notebooks", "s2_coint", "02_ev_vs_spy.ipynb")


def _synthetic_frame(n: int, *, freq: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if freq == "W":
        idx = pd.bdate_range("2016-01-04", periods=n, freq="W-MON")
    else:
        idx = pd.bdate_range("2019-01-02", periods=n, freq="B")
    spy = rng.normal(0.0015, 0.016, n)
    strat = 0.0008 + 0.3 * spy + rng.normal(0, 0.012, n)
    return pd.DataFrame({"strategy": strat, "spy": spy}, index=idx)


def _assign_frame_cell(frame: pd.DataFrame, parquet_path: str) -> str:
    frame.to_parquet(parquet_path)
    return (
        "import pandas as pd\n"
        f"FRAME = pd.read_parquet(r{parquet_path!r})\n"
        "FRAME.index = pd.to_datetime(FRAME.index)\n"
        "print('synthetic FRAME', len(FRAME), FRAME.index.min().date(), FRAME.index.max().date())\n"
    )


def _prepare_notebook(
    path: str, frame: pd.DataFrame, parquet_path: str
) -> nbformat.NotebookNode:
    nb = nbformat.read(path, as_version=4)
    cfg = "".join(nb.cells[2].source)
    cfg = cfg.replace("DEFAULT_N_SIM = 400", "DEFAULT_N_SIM = 20")
    if "DEFAULT_H = 52" in cfg:
        cfg = cfg.replace("DEFAULT_H = 52", "DEFAULT_H = 10")
    if "DEFAULT_H = 63" in cfg:
        cfg = cfg.replace("DEFAULT_H = 63", "DEFAULT_H = 10")
    if "N_BOOTSTRAP = 600" in cfg:
        cfg = cfg.replace("N_BOOTSTRAP = 600", "N_BOOTSTRAP = 40")
    nb.cells[2].source = cfg
    nb.cells[4].source = _assign_frame_cell(frame, parquet_path)
    if len(nb.cells) != 15:
        # S1 (legacy widget MC cell)
        run = "".join(nb.cells[9].source)
        run = run.replace("N_BOOTSTRAP = 600", "N_BOOTSTRAP = 40")
        idx = run.find("\ntry:\n")
        if idx >= 0:
            run = run[: idx + 1]
        nb.cells[9].source = run
    hmm = "".join(nb.cells[-1].source)
    hmm = hmm.replace("n_simulations=200", "n_simulations=12")
    nb.cells[-1].source = hmm
    return nb


def _execute(nb: nbformat.NotebookNode) -> None:
    pytest.importorskip("nbclient")
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import NoSuchKernel

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


def test_s1_ev_notebook_executes(tmp_path):
    frame = _synthetic_frame(36, freq="W", seed=5)
    nb = _prepare_notebook(S1_NB, frame, os.path.join(str(tmp_path), "s1.parquet"))
    _execute(nb)
    # last geometry code cell should have run; PACK exists in a prior cell
    sources = ["".join(c.source) for c in nb.cells]
    assert any("Pathwise holes" in s for s in sources)
    assert any("Joint shape vs SPY" in s for s in sources)


def test_s2_ev_notebook_executes(tmp_path):
    frame = _synthetic_frame(40, freq="D", seed=6)
    nb = _prepare_notebook(S2_NB, frame, os.path.join(str(tmp_path), "s2.parquet"))
    _execute(nb)
    sources = ["".join(c.source) for c in nb.cells]
    assert any("Joint paths vs SPY" in s for s in sources)
    assert any("Pathwise holes" in s for s in sources)
    assert any("excess_fan" in s for s in sources)
    assert any("run_ev_vs_spy" in s for s in sources)
