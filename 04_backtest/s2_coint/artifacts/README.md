# S2 backtest artifacts

Research-only. Notebooks import `backtest.s2_coint.runner` — there is **no** `s2_strategy.py` in this program.

- `s2_star_stack.json` is written **only** from a freeze cell where you **type** the STAR constant after reviewing fold-val metrics. Notebooks must not assign STAR from `argmax`.
- Run order: `H-001_universes.ipynb` (DECIDED) → `s2_pair_panel.ipynb` (1D) → `H-002_bar_size.ipynb` (1D vs 1H, **no 4H**) → `H-003` … `H-013`. Never re-open an earlier STAR.
- Tearsheet PDFs: `H-00X_{arm}.pdf`.
- Parquet panels stay under `01_data/data_files/s2_coint/` (gitignored).
