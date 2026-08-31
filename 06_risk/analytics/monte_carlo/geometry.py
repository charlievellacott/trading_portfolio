"""Pathwise holes and joint risk geometry vs SPY (EV package only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk.analytics.monte_carlo.ev_stats import cvar, path_wealth, terminal_simple_return


def _align_path_frames(
    strategy_paths: pd.DataFrame,
    spy_paths: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = strategy_paths.astype(float)
    b = spy_paths.astype(float)
    cols = s.columns.intersection(b.columns)
    if len(cols) == 0:
        raise ValueError("strategy and SPY path columns do not overlap (need joint paths)")
    return s.loc[:, cols], b.loc[:, cols]


def wealth_with_start(paths: pd.DataFrame) -> pd.DataFrame:
    """Wealth including the implicit 1.0 before the first return."""
    w = path_wealth(paths)
    start = pd.DataFrame(1.0, index=[-1], columns=w.columns)
    out = pd.concat([start, w], axis=0)
    out.index = range(out.shape[0])
    return out


def path_max_drawdown(paths: pd.DataFrame) -> pd.Series:
    """Peak-to-trough drawdown per path (negative), measured from start wealth 1.0."""
    w = wealth_with_start(paths).to_numpy(dtype=float)
    peak = np.maximum.accumulate(w, axis=0)
    dd = w / np.clip(peak, 1e-15, None) - 1.0
    out = pd.Series(dd.min(axis=0), index=paths.columns, name="max_dd")
    return out


def path_frac_in_drawdown(paths: pd.DataFrame) -> pd.Series:
    """Fraction of bars (after start) spent below the running peak."""
    w = wealth_with_start(paths).to_numpy(dtype=float)
    peak = np.maximum.accumulate(w, axis=0)
    in_dd = w < peak - 1e-15
    # skip the leading 1.0 row
    frac = in_dd[1:].mean(axis=0)
    return pd.Series(frac, index=paths.columns, name="frac_in_drawdown")


def path_frac_below_start(paths: pd.DataFrame) -> pd.Series:
    """Fraction of post-start bars with wealth < 1."""
    w = path_wealth(paths)
    return (w < 1.0).mean(axis=0).rename("frac_below_start")


def path_bars_to_recover(paths: pd.DataFrame) -> pd.Series:
    """Bars from the max-DD trough until wealth retouches that peak; NaN if never."""
    w = wealth_with_start(paths).to_numpy(dtype=float)
    peak = np.maximum.accumulate(w, axis=0)
    dd = w / np.clip(peak, 1e-15, None) - 1.0
    n_bar, n_path = w.shape
    rec = np.full(n_path, np.nan, dtype=float)
    for j in range(n_path):
        trough = int(np.argmin(dd[:, j]))
        hw = peak[trough, j]
        later = w[trough:, j]
        hit = np.flatnonzero(later >= hw - 1e-12)
        if hit.size:
            rec[j] = float(hit[0])
    return pd.Series(rec, index=paths.columns, name="bars_to_recover")


def pathwise_holes(paths: pd.DataFrame) -> pd.DataFrame:
    """Per-path hole metrics plus terminal wealth (for the DD scatter)."""
    term_w = path_wealth(paths).iloc[-1]
    holes = pd.DataFrame(
        {
            "terminal_wealth": term_w.astype(float),
            "terminal_return": term_w.astype(float) - 1.0,
            "max_dd": path_max_drawdown(paths),
            "frac_in_drawdown": path_frac_in_drawdown(paths),
            "frac_below_start": path_frac_below_start(paths),
            "bars_to_recover": path_bars_to_recover(paths),
        }
    )
    return holes


def holes_summary(holes: pd.DataFrame) -> pd.Series:
    """Desk percentiles for pathwise holes (not P(ever underwater))."""
    if holes.empty:
        return pd.Series(dtype=float, name="holes")
    dd = holes["max_dd"]
    rec = holes["bars_to_recover"]
    return pd.Series(
        {
            "max_dd_mean": float(dd.mean()),
            "max_dd_median": float(dd.median()),
            "max_dd_p10": float(dd.quantile(0.10)),
            "max_dd_p05": float(dd.quantile(0.05)),
            "frac_in_dd_median": float(holes["frac_in_drawdown"].median()),
            "frac_below_start_median": float(holes["frac_below_start"].median()),
            "bars_to_recover_median": float(rec.median()) if rec.notna().any() else float("nan"),
            "bars_to_recover_p90": float(rec.quantile(0.90)) if rec.notna().any() else float("nan"),
            "p_never_recover": float(rec.isna().mean()),
            "n_paths": float(len(holes)),
        },
        name="holes",
    )


def ev_concentration(paths: pd.DataFrame, *, top_frac: float = 0.10) -> pd.Series:
    """Mean vs median vs CVaR of terminals; share of total EV from the best paths."""
    term = terminal_simple_return(paths)
    n = int(term.shape[0])
    if n == 0:
        return pd.Series(dtype=float, name="concentration")
    k = max(1, int(round(float(top_frac) * n)))
    top = term.nlargest(k)
    total = float(term.sum())
    share = float(top.sum() / total) if total != 0.0 else float("nan")
    return pd.Series(
        {
            "mean_terminal": float(term.mean()),
            "median_terminal": float(term.median()),
            "cvar_5": cvar(term, alpha=0.05),
            "top_decile_ev_share": share,
            "top_n": float(k),
        },
        name="concentration",
    )


def _ols_beta_corr(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    if y.size < 5:
        return float("nan"), float("nan")
    var_x = float(np.var(x, ddof=1))
    if var_x <= 0.0:
        return float("nan"), float("nan")
    cov = float(np.cov(y, x, ddof=1)[0, 1])
    beta = cov / var_x
    sd_y = float(np.std(y, ddof=1))
    sd_x = float(np.std(x, ddof=1))
    corr = cov / (sd_y * sd_x) if sd_y > 0.0 and sd_x > 0.0 else float("nan")
    return beta, corr


def _nanmed(x: np.ndarray) -> float:
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("nan")
    return float(np.median(y))


def joint_shape_vs_spy(
    strategy_paths: pd.DataFrame,
    spy_paths: pd.DataFrame,
) -> pd.Series:
    """Beta/corr, down-market capture, and P(not beat | SPY finished underwater).

    Uses the same paired columns (joint bootstrap). Independent reshuffle is invalid.
    """
    s, b = _align_path_frames(strategy_paths, spy_paths)
    n_paths = int(s.shape[1])
    betas = np.full(n_paths, np.nan)
    corrs = np.full(n_paths, np.nan)
    captures = np.full(n_paths, np.nan)
    s_arr = s.to_numpy(dtype=float)
    b_arr = b.to_numpy(dtype=float)
    for j in range(n_paths):
        betas[j], corrs[j] = _ols_beta_corr(s_arr[:, j], b_arr[:, j])
        down = b_arr[:, j] < 0.0
        if down.any():
            captures[j] = float(s_arr[down, j].mean())
    w_s = path_wealth(s).iloc[-1]
    w_b = path_wealth(b).iloc[-1]
    spy_uw = w_b.to_numpy() < 1.0
    if spy_uw.any():
        p_lose_given_spy_uw = float((w_s.to_numpy()[spy_uw] <= w_b.to_numpy()[spy_uw]).mean())
    else:
        p_lose_given_spy_uw = float("nan")
    p_spy_uw = float(spy_uw.mean())
    return pd.Series(
        {
            "beta_median": _nanmed(betas),
            "corr_median": _nanmed(corrs),
            "down_capture_median": _nanmed(captures),
            "p_not_beat_given_spy_underwater": p_lose_given_spy_uw,
            "p_spy_terminal_underwater": p_spy_uw,
            "n_paths": float(n_paths),
        },
        name="joint_shape",
    )


def excess_wealth_paths(
    strategy_paths: pd.DataFrame,
    spy_paths: pd.DataFrame,
) -> pd.DataFrame:
    """``W_strategy / W_spy`` per bar (joint columns)."""
    s, b = _align_path_frames(strategy_paths, spy_paths)
    return path_wealth(s) / path_wealth(b).replace(0.0, np.nan)


def realized_window_returns(returns: pd.Series, horizon: int) -> pd.Series:
    """Most recent ``horizon`` bars (or the full series if shorter)."""
    s = pd.to_numeric(returns, errors="coerce").astype(float).dropna()
    h = max(1, int(horizon))
    if s.empty:
        return s
    return s.iloc[-h:]


def realized_wealth_for_fan(returns: pd.Series, horizon: int) -> pd.Series:
    """Wealth path (no leading 1.0) aligned to fan bars ``0 .. len-1``."""
    window = realized_window_returns(returns, horizon)
    if window.empty:
        return pd.Series(dtype=float, name="realized")
    w = (1.0 + window).cumprod()
    out = pd.Series(w.to_numpy(dtype=float), index=range(len(w)), name="realized")
    return out


def realized_first_window_wealth(returns: pd.Series, horizon: int) -> pd.Series | None:
    """First ``horizon`` bars of OOS, if the sample is longer than H."""
    s = pd.to_numeric(returns, errors="coerce").astype(float).dropna()
    h = max(1, int(horizon))
    if len(s) <= h:
        return None
    w = (1.0 + s.iloc[:h]).cumprod()
    return pd.Series(w.to_numpy(dtype=float), index=range(h), name="realized_first")


def realized_terminal_percentile(
    realized_wealth: pd.Series,
    simulated_paths: pd.DataFrame,
) -> float:
    """Percentile of realized terminal among simulated terminals (0–100)."""
    if realized_wealth is None or realized_wealth.empty:
        return float("nan")
    h = int(realized_wealth.shape[0])
    if simulated_paths.shape[0] < h:
        return float("nan")
    sim_term = path_wealth(simulated_paths.iloc[:h]).iloc[-1]
    r_term = float(realized_wealth.iloc[-1])
    return float((sim_term.to_numpy() <= r_term).mean() * 100.0)
