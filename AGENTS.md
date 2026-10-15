# AGENTS.md

## Cursor Cloud specific instructions

This is a Python (>=3.11) quantitative trading research/portfolio codebase with numbered
top-level directories (`01_data` … `10_tests`). The startup update script already runs the
dependency install, so you normally do not need to reinstall anything.

### Layout / imports (non-obvious)
- Numbered dirs cannot be Python packages, so `pyproject.toml` remaps them to logical names
  (`data`, `models`, `backtest`, `strategies`, `risk`, `execution`, `portfolio`,
  `performance`) via an **editable install**. `conftest.py` hard-fails collection if that
  editable install is missing, so pytest will refuse to run until
  `pip install -e .` has succeeded (handled by the update script).
- Local imports must use absolute `from data...`/`from execution...` form (see
  `.cursor/rules/import-style.mdc`).

### Running tests
- Tests live in **`10_tests/`** (the `README.md` reference to `08_tests/` is stale).
- Run the suite with `python -m pytest 10_tests/ -q` (225 tests, ~45s).
- The suite includes **live-network integration tests** (Yahoo Finance via `yfinance`,
  SEC EDGAR, FINRA) with no offline markers, so a full green run needs internet. Pure-logic
  suites like `test_pct_stop.py`, `test_vol_targeting.py`, and the `s2_coint` tests run offline.
- `pytest`/other console scripts install to `~/.local/bin` (not on `PATH`); invoke via
  `python -m pytest` rather than the bare `pytest` binary.

### Running the application (S1 equities is the mature sleeve)
- Fast offline smoke test of the real order-placement runner (no creds, no network):
  `python 10_tests/s1_equities/s1_pipeline_tester.py` — exercises `place_orders`, GTC stop
  placement, uncover-close, and liquidation against a `FakeBroker`.
- `--live-strategy` runs the real `S1Strategy` (live fetch → features → weights). It is
  **slow** (pulls ~79 months of FINRA short-volume + GDELT data) and requires **Google Cloud
  BigQuery Application Default Credentials** for the GDELT sentiment factor; without ADC it
  fails at the sentiment step. FINRA archive endpoints also intermittently return HTTP 403 on
  recent dates.
- `--paper` and `07_execution/s1_equities/s1_paper_runner.py` need **Alpaca paper keys** in
  `config/credentials.env` (gitignored, absent by default; key order per `README.md`:
  Alpaca key, Alpaca secret, OANDA key). The paper runner also only acts Monday during NYSE RTH.

### Other notes
- No linter/formatter/type-checker is configured (no ruff/flake8/black/pylint/mypy config or
  deps; `basedpyright` is only referenced in a `pyproject.toml` comment). There is no lint step.
- No parquet panels or caches are committed (all `*.parquet` are gitignored). Feature panels are
  built on demand from live data or notebooks. Caches default under `01_data/cache/` and
  `09_performance/cache/`; the S1 cache dir can be overridden with `S1_CACHE_DIR`.
- `TA-Lib` requires the native `ta-lib` C library (system dependency, baked into the image, not
  in the update script).
