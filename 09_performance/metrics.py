"""Portfolio risk metrics backed by vectorbt."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import vectorbt  # registers pandas .vbt accessor

_RATIO_COLOR = "#1f4e79"
_DRAWDOWN_COLOR = "#a01313"
_GRID_COLOR = "#d9dde3"
_ZERO_LINE_COLOR = "#8a9199"
_REFERENCE_LINE_COLOR = "#b8bcc4"

_METRIC_LABELS = {
    "sharpe": "Rolling Sharpe Ratio",
    "sortino": "Rolling Sortino Ratio",
    "calmar": "Rolling Calmar Ratio",
    "max_drawdown": "Rolling Max Drawdown",
}


def as_return_series(data: Any, *, from_equity: bool = False) -> pd.Series:
    """
    Normalize portfolio outputs to a single period-return Series.

    Accepts a vectorbt Portfolio, return Series/DataFrame, or equity curve
    (when ``from_equity=True``).
    """
    if hasattr(data, "returns"):
        series = data.returns
        if isinstance(series, pd.DataFrame):
            if series.shape[1] != 1:
                raise ValueError("portfolio returns must be a single column")
            series = series.iloc[:, 0]
        elif not isinstance(series, pd.Series):
            raise TypeError(f"unsupported portfolio returns type: {type(series)!r}")
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("returns DataFrame must have exactly one column")
        series = data.iloc[:, 0]
    elif isinstance(data, pd.Series):
        series = data
    else:
        raise TypeError(
            f"expected Portfolio, pd.Series, or 1-column pd.DataFrame; got {type(data)!r}"
        )

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("returns must have a DatetimeIndex")

    if from_equity:
        series = series.pct_change()

    series = series.dropna()
    if series.empty:
        raise ValueError("returns series is empty after normalization")

    return series


def _returns_accessor(
    data: Any,
    *,
    freq: str,
    year_freq: str,
    risk_free: float,
    from_equity: bool,
):
    series = as_return_series(data, from_equity=from_equity)
    defaults = {"risk_free": risk_free}
    return series.vbt.returns(freq=freq, year_freq=year_freq, defaults=defaults)


def sharpe_ratio(
    returns: Any,
    *,
    freq: str = "d",
    year_freq: str = "252d",
    risk_free: float = 0.0,
    from_equity: bool = False,
) -> float:
    """Annualized Sharpe ratio from period returns."""
    ret_acc = _returns_accessor(
        returns,
        freq=freq,
        year_freq=year_freq,
        risk_free=risk_free,
        from_equity=from_equity,
    )
    return float(ret_acc.sharpe_ratio())


def sortino_ratio(
    returns: Any,
    *,
    freq: str = "d",
    year_freq: str = "252d",
    risk_free: float = 0.0,
    from_equity: bool = False,
) -> float:
    """Annualized Sortino ratio from period returns."""
    ret_acc = _returns_accessor(
        returns,
        freq=freq,
        year_freq=year_freq,
        risk_free=risk_free,
        from_equity=from_equity,
    )
    return float(ret_acc.sortino_ratio())


def calmar_ratio(
    returns: Any,
    *,
    freq: str = "d",
    year_freq: str = "252d",
    from_equity: bool = False,
) -> float:
    """Annualized Calmar ratio from period returns."""
    ret_acc = _returns_accessor(
        returns,
        freq=freq,
        year_freq=year_freq,
        risk_free=0.0,
        from_equity=from_equity,
    )
    return float(ret_acc.calmar_ratio())


def max_drawdown(
    returns: Any,
    *,
    freq: str = "d",
    from_equity: bool = False,
) -> float:
    """Maximum drawdown as a negative fraction (vectorbt convention)."""
    series = as_return_series(returns, from_equity=from_equity)
    ret_acc = series.vbt.returns(freq=freq)
    return float(ret_acc.max_drawdown())


def rolling_sharpe_ratio(
    returns: Any,
    lookback: int,
    *,
    freq: str = "d",
    year_freq: str = "252d",
    risk_free: float = 0.0,
    from_equity: bool = False,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling annualized Sharpe ratio over ``lookback`` bars."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    ret_acc = _returns_accessor(
        returns,
        freq=freq,
        year_freq=year_freq,
        risk_free=risk_free,
        from_equity=from_equity,
    )
    return ret_acc.rolling_sharpe_ratio(window=lookback, minp=min_periods)


def rolling_sortino_ratio(
    returns: Any,
    lookback: int,
    *,
    freq: str = "d",
    year_freq: str = "252d",
    risk_free: float = 0.0,
    from_equity: bool = False,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling annualized Sortino ratio over ``lookback`` bars."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    ret_acc = _returns_accessor(
        returns,
        freq=freq,
        year_freq=year_freq,
        risk_free=risk_free,
        from_equity=from_equity,
    )
    return ret_acc.rolling_sortino_ratio(window=lookback, minp=min_periods)


def rolling_calmar_ratio(
    returns: Any,
    lookback: int,
    *,
    freq: str = "d",
    year_freq: str = "252d",
    from_equity: bool = False,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling annualized Calmar ratio over ``lookback`` bars."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    ret_acc = _returns_accessor(
        returns,
        freq=freq,
        year_freq=year_freq,
        risk_free=0.0,
        from_equity=from_equity,
    )
    return ret_acc.rolling_calmar_ratio(window=lookback, minp=min_periods)


def rolling_max_drawdown(
    returns: Any,
    lookback: int,
    *,
    freq: str = "d",
    from_equity: bool = False,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling maximum drawdown over ``lookback`` bars (negative fraction)."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    series = as_return_series(returns, from_equity=from_equity)
    ret_acc = series.vbt.returns(freq=freq)
    return ret_acc.rolling_max_drawdown(window=lookback, minp=min_periods)


_ROLLING_DISPATCH = {
    "sharpe": rolling_sharpe_ratio,
    "sortino": rolling_sortino_ratio,
    "calmar": rolling_calmar_ratio,
    "max_drawdown": rolling_max_drawdown,
}


@contextmanager
def _plot_style_context():
    for style_name in ("seaborn-v0_8-whitegrid", "ggplot"):
        if style_name in plt.style.available:
            with plt.style.context(style_name):
                yield
            return
    yield


def _style_axis(ax: plt.Axes, *, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_GRID_COLOR)
    ax.spines["bottom"].set_color(_GRID_COLOR)
    ax.tick_params(axis="both", labelsize=9, colors="#4a4f57")
    ax.grid(True, axis="y", color=_GRID_COLOR, linewidth=0.8, alpha=0.7)
    ax.grid(False, axis="x")


def _plot_ratio_series(
    ax: plt.Axes,
    series: pd.Series,
    *,
    title: str,
    show_one_reference: bool,
) -> None:
    ax.plot(series.index, series.values, color=_RATIO_COLOR, linewidth=1.6)
    ax.axhline(0.0, color=_ZERO_LINE_COLOR, linewidth=0.9, linestyle="-", zorder=1)
    if show_one_reference:
        ax.axhline(
            1.0,
            color=_REFERENCE_LINE_COLOR,
            linewidth=0.9,
            linestyle="--",
            zorder=1,
        )
    _style_axis(ax, ylabel="Ratio")
    ax.set_title(title, fontsize=11, fontweight="semibold", loc="left", pad=10)


def _plot_drawdown_series(ax: plt.Axes, series: pd.Series, *, title: str) -> None:
    values = series.values
    ax.fill_between(series.index, values, 0.0, color=_DRAWDOWN_COLOR, alpha=0.18)
    ax.plot(series.index, values, color=_DRAWDOWN_COLOR, linewidth=1.6)
    ax.axhline(0.0, color=_ZERO_LINE_COLOR, linewidth=0.9, linestyle="-", zorder=1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    _style_axis(ax, ylabel="Drawdown")
    ax.set_title(title, fontsize=11, fontweight="semibold", loc="left", pad=10)


def plot_rolling_metrics(
    returns: Any,
    lookback: int,
    *,
    metrics: Sequence[str] = ("sharpe", "sortino", "calmar", "max_drawdown"),
    freq: str = "d",
    year_freq: str = "252d",
    risk_free: float = 0.0,
    from_equity: bool = False,
    min_periods: int | None = None,
    title: str | None = None,
) -> None:
    """
    Display rolling risk metrics, one subplot per metric.

    Calls ``plt.show()`` and returns ``None``.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if not metrics:
        raise ValueError("metrics must contain at least one metric name")

    unknown = [name for name in metrics if name not in _ROLLING_DISPATCH]
    if unknown:
        raise ValueError(f"unknown metrics: {unknown}; expected keys in {_ROLLING_DISPATCH}")

    rolling_series: dict[str, pd.Series] = {}
    for name in metrics:
        kwargs = {
            "freq": freq,
            "from_equity": from_equity,
            "min_periods": min_periods,
        }
        if name != "max_drawdown":
            kwargs["year_freq"] = year_freq
        if name in {"sharpe", "sortino"}:
            kwargs["risk_free"] = risk_free
        rolling_series[name] = _ROLLING_DISPATCH[name](returns, lookback, **kwargs)

    n_metrics = len(metrics)
    with _plot_style_context():
        fig, axes = plt.subplots(
            n_metrics,
            1,
            sharex=True,
            figsize=(11.0, 2.6 * n_metrics),
            constrained_layout=True,
        )
        if n_metrics == 1:
            axes = [axes]

        if title:
            fig.suptitle(title, fontsize=13, fontweight="semibold", y=1.02)

        for ax, name in zip(axes, metrics):
            series = rolling_series[name]
            plot_title = f"{_METRIC_LABELS[name]} (lookback={lookback})"
            if name == "max_drawdown":
                _plot_drawdown_series(ax, series, title=plot_title)
            else:
                _plot_ratio_series(
                    ax,
                    series,
                    title=plot_title,
                    show_one_reference=name in {"sharpe", "sortino"},
                )

        axes[-1].set_xlabel("Date", fontsize=10, labelpad=8)
        fig.autofmt_xdate()
        plt.show()
