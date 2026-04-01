"""Fundamentals model for midterm forecasting.

This module implements a structural prior component that answers: *given the
current economic and political environment, what should we expect to happen in
November, before any race-specific polls are considered?*

The model fits a Ridge regression on historical midterm cycles (1974–2022,
13 data points) using four predictors:

    fundamentals_shift_pp = (
        beta_approval  * pres_net_approval_oct
        + beta_gdp     * gdp_q2_growth_pct
        + beta_unemp   * unemployment_oct
        + beta_cpi     * cpi_yoy_oct
        + intercept
    )

With only ~13 data points, Ridge regularization is **essential**.  The model
deliberately has wide uncertainty (LOO RMSE ~2–3pp) — this is expected and
correct; the fundamentals layer is one prior among several, not a standalone
prediction.

Typical usage:

    from src.prediction.fundamentals import (
        load_fundamentals_snapshot,
        compute_fundamentals_shift,
        apply_fundamentals_shift,
    )

    snapshot = load_fundamentals_snapshot()
    info = compute_fundamentals_shift(snapshot)
    shifted_priors = apply_fundamentals_shift(county_priors, info.shift)

The output ``info.shift`` is in the same Dem-share units as the generic ballot
shift in ``generic_ballot.py``, so it slots directly into the forecast pipeline.

Historical calibration data lives at:
    ``data/raw/fundamentals/midterm_history.csv``

Current-cycle snapshot lives at:
    ``data/fundamentals/snapshot_2026.json``
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default paths — can be overridden in tests.
_DEFAULT_HISTORY_PATH = PROJECT_ROOT / "data" / "raw" / "fundamentals" / "midterm_history.csv"
_DEFAULT_SNAPSHOT_PATH = PROJECT_ROOT / "data" / "fundamentals" / "snapshot_2026.json"

# Clamp shifted priors to this range to keep values in valid probability range.
_PRIOR_MIN: float = 0.01
_PRIOR_MAX: float = 0.99

# Ridge regularization strength.  Deliberately high given N~13.
# At alpha=10 the model is meaningfully regularized but still tracks the data.
_DEFAULT_RIDGE_ALPHA: float = 10.0

# Minimum historical cycles needed to fit the four-predictor Ridge model.
_MIN_TRAINING_RECORDS: int = 4

# Convert Ridge output (percentage points) to Dem-share fraction for use in
# the county prior pipeline (same units as generic ballot shift).
_PP_TO_FRACTION: float = 0.01


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FundamentalsSnapshot:
    """Raw inputs for a single election cycle's fundamentals model.

    All numeric fields use the same units as the historical calibration CSV so
    the fitted Ridge coefficients apply directly.

    Attributes
    ----------
    cycle:
        Election year (e.g. 2026).
    in_party:
        President's party: "D" or "R".
    approval_net_oct:
        Presidential net approval rating in October of the election year
        (approve% - disapprove%).  Negative means underwater.
    gdp_q2_growth_pct:
        Real GDP quarter-over-quarter annualized growth in Q2 of the election
        year (percent, e.g. 1.8 for +1.8%).
    unemployment_oct:
        National unemployment rate in October of the election year (percent,
        e.g. 4.1 for 4.1%).
    cpi_yoy_oct:
        Year-over-year CPI inflation in October of the election year (percent,
        e.g. 3.2 for 3.2%).
    consumer_sentiment:
        University of Michigan Consumer Sentiment Index (optional; not used in
        the v1 Ridge model but stored for future use).
    source_notes:
        Free-form dict with data provenance strings for each field.
    """

    cycle: int
    in_party: str  # "D" or "R"
    approval_net_oct: float
    gdp_q2_growth_pct: float
    unemployment_oct: float
    cpi_yoy_oct: float
    consumer_sentiment: Optional[float] = None
    source_notes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FundamentalsInfo:
    """Result of a fundamentals model computation.

    Analogous to ``GenericBallotInfo`` from ``generic_ballot.py``.  The
    ``shift`` field is in the same Dem-share units and can be passed directly
    to ``apply_fundamentals_shift()`` or ``apply_gb_shift()``.

    Attributes
    ----------
    shift:
        Model-predicted national Dem share change relative to the 2024
        presidential baseline (positive = Dems doing better than 2024).
    approval_contribution:
        Component of shift attributed to presidential approval.
    gdp_contribution:
        Component of shift attributed to GDP growth.
    unemployment_contribution:
        Component of shift attributed to unemployment.
    cpi_contribution:
        Component of shift attributed to CPI inflation.
    intercept_contribution:
        Baseline in-party penalty from the regression intercept.
    loo_rmse:
        Leave-one-out root mean squared error from the fitted model (pp).
        Represents expected out-of-sample error magnitude.
    n_training:
        Number of historical cycles used to fit the model.
    source:
        Human-readable description ("fitted_ridge" or "fallback").
    """

    shift: float
    approval_contribution: float
    gdp_contribution: float
    unemployment_contribution: float
    cpi_contribution: float
    intercept_contribution: float
    loo_rmse: float
    n_training: int
    source: str


# ---------------------------------------------------------------------------
# Historical data loading
# ---------------------------------------------------------------------------


@dataclass
class _HistoricalRecord:
    """One row from midterm_history.csv, parsed into typed fields."""

    year: int
    pres_party: str
    pres_net_approval_oct: float
    gdp_q2_growth_pct: float
    unemployment_oct: float
    cpi_yoy_oct: float
    dem_house_share_change_pp: float


_HISTORY_REQUIRED_COLS = {
    "year", "pres_party", "pres_net_approval_oct",
    "gdp_q2_growth_pct", "unemployment_oct", "cpi_yoy_oct",
    "dem_house_share_change_pp",
}


def _validate_history_csv_columns(reader: csv.DictReader) -> None:
    """Raise ValueError if the CSV is empty or missing required columns."""
    if reader.fieldnames is None:
        raise ValueError("CSV appears to be empty")
    missing = _HISTORY_REQUIRED_COLS - set(reader.fieldnames)
    if missing:
        raise ValueError(f"midterm_history.csv missing columns: {missing}")


def _parse_history_row(row: dict) -> Optional[_HistoricalRecord]:
    """Parse one CSV row into a _HistoricalRecord; return None and log on error."""
    try:
        return _HistoricalRecord(
            year=int(row["year"]),
            pres_party=row["pres_party"].strip(),
            pres_net_approval_oct=float(row["pres_net_approval_oct"]),
            gdp_q2_growth_pct=float(row["gdp_q2_growth_pct"]),
            unemployment_oct=float(row["unemployment_oct"]),
            cpi_yoy_oct=float(row["cpi_yoy_oct"]),
            dem_house_share_change_pp=float(row["dem_house_share_change_pp"]),
        )
    except (ValueError, KeyError) as exc:
        log.warning("Skipping malformed row in midterm_history.csv (year=%s): %s", row.get("year"), exc)
        return None


def load_historical_data(
    history_path: Path | str | None = None,
) -> list[_HistoricalRecord]:
    """Load and parse the midterm history calibration CSV.

    Parameters
    ----------
    history_path:
        Path to ``midterm_history.csv``.  Defaults to
        ``data/raw/fundamentals/midterm_history.csv`` relative to project root.

    Returns
    -------
    list[_HistoricalRecord]
        Parsed records, sorted by year ascending.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the resolved path.
    ValueError
        If any required column is missing from the header.
    """
    if history_path is None:
        history_path = _DEFAULT_HISTORY_PATH
    history_path = Path(history_path)

    if not history_path.exists():
        raise FileNotFoundError(
            f"Midterm history CSV not found at {history_path}. "
            "Expected columns: year, pres_party, pres_net_approval_oct, "
            "gdp_q2_growth_pct, unemployment_oct, cpi_yoy_oct, dem_house_share_change_pp"
        )

    records: list[_HistoricalRecord] = []
    with history_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _validate_history_csv_columns(reader)
        for row in reader:
            rec = _parse_history_row(row)
            if rec is not None:
                records.append(rec)

    records.sort(key=lambda r: r.year)
    log.debug("Loaded %d historical midterm records from %s", len(records), history_path)
    return records


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class FundamentalsModel:
    """Ridge regression model fitted on historical midterm cycles.

    Predicts Dem House share change (pp) from four national fundamentals:
    presidential approval, GDP Q2 growth, unemployment, and CPI inflation.

    With only ~13 training points, the model is deliberately parsimonious
    and heavily regularized.  Coefficients are interpretable as direction
    and rough magnitude — not precise point estimates.

    Parameters
    ----------
    alpha:
        Ridge regularization strength.  Higher values shrink coefficients
        toward zero more aggressively.  Default 10.0 is conservative given N~13.

    Attributes
    ----------
    coef_:
        Fitted coefficients [approval, gdp, unemployment, cpi].
    intercept_:
        Fitted intercept (in-party structural penalty/bonus, pp).
    loo_rmse_:
        Leave-one-out RMSE on the training data (pp).  Use this as the
        uncertainty estimate — *not* in-sample residuals.
    n_training_:
        Number of observations used for fitting.
    is_fitted_:
        True after ``fit()`` has been called successfully.
    """

    def __init__(self, alpha: float = _DEFAULT_RIDGE_ALPHA) -> None:
        self.alpha = alpha
        self.coef_: np.ndarray = np.zeros(4)
        self.intercept_: float = 0.0
        self.loo_rmse_: float = float("nan")
        self.n_training_: int = 0
        self.is_fitted_: bool = False

    def _build_feature_matrix(
        self,
        records: list[_HistoricalRecord],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract X (N×4) and y (N,) from historical records.

        Features: [approval, gdp_q2_growth, unemployment, cpi_yoy]
        Target: dem_house_share_change_pp
        """
        X = np.array([
            [r.pres_net_approval_oct, r.gdp_q2_growth_pct,
             r.unemployment_oct, r.cpi_yoy_oct]
            for r in records
        ], dtype=float)
        y = np.array([r.dem_house_share_change_pp for r in records], dtype=float)
        return X, y

    @staticmethod
    def _ridge_fit(
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, float]:
        """Fit Ridge regression: (XᵀX + αI)⁻¹ Xᵀy with mean-centering.

        Returns
        -------
        coef : shape (p,)
        intercept : float
        """
        # Mean-center X for numeric stability; intercept absorbs the mean of y.
        X_mean = X.mean(axis=0)
        y_mean = float(y.mean())
        X_c = X - X_mean
        y_c = y - y_mean

        n, p = X_c.shape
        A = X_c.T @ X_c + alpha * np.eye(p)
        coef = np.linalg.solve(A, X_c.T @ y_c)
        intercept = y_mean - X_mean @ coef
        return coef, intercept

    def _compute_loo_rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute leave-one-out RMSE as the model's uncertainty estimate.

        LOO is used instead of in-sample residuals because N~13 makes in-sample
        metrics wildly optimistic.  Each fold trains on N-1 points and predicts
        the held-out point; RMSE is taken over all N residuals.
        """
        n = len(y)
        loo_residuals: list[float] = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_train, y_train = X[mask], y[mask]
            X_test, y_test = X[i : i + 1], y[i]
            coef_i, intercept_i = self._ridge_fit(X_train, y_train, self.alpha)
            y_pred = float((X_test @ coef_i).item() + intercept_i)
            loo_residuals.append(y_pred - float(y_test))
        return float(np.sqrt(np.mean(np.array(loo_residuals) ** 2)))

    def fit(self, records: list[_HistoricalRecord]) -> "FundamentalsModel":
        """Fit the Ridge model on historical records and compute LOO RMSE.

        Parameters
        ----------
        records:
            Output of ``load_historical_data()``.  Must have at least
            ``_MIN_TRAINING_RECORDS`` observations (one per feature); more is
            strongly recommended.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If fewer than ``_MIN_TRAINING_RECORDS`` records are provided.
        """
        if len(records) < _MIN_TRAINING_RECORDS:
            raise ValueError(
                f"Need at least {_MIN_TRAINING_RECORDS} historical midterm cycles to fit; "
                f"got {len(records)}."
            )

        X, y = self._build_feature_matrix(records)
        n = len(records)

        self.coef_, self.intercept_ = self._ridge_fit(X, y, self.alpha)
        self.n_training_ = n
        self.loo_rmse_ = self._compute_loo_rmse(X, y)
        self.is_fitted_ = True

        log.info(
            "FundamentalsModel fitted on %d cycles: "
            "coef(approval=%.3f, gdp=%.3f, unemp=%.3f, cpi=%.3f) "
            "intercept=%.3f LOO_RMSE=%.2f pp",
            n, self.coef_[0], self.coef_[1], self.coef_[2], self.coef_[3],
            self.intercept_, self.loo_rmse_,
        )
        return self

    def predict(
        self,
        approval_net: float,
        gdp_q2_growth: float,
        unemployment: float,
        cpi_yoy: float,
    ) -> tuple[float, float, float, float, float]:
        """Predict Dem House share change and per-component contributions.

        Parameters
        ----------
        approval_net:
            Presidential net approval (pp).
        gdp_q2_growth:
            Q2 real GDP growth (%).
        unemployment:
            Unemployment rate (%).
        cpi_yoy:
            Year-over-year CPI inflation (%).

        Returns
        -------
        total, approval_contrib, gdp_contrib, unemp_contrib, cpi_contrib : floats
            Predicted total shift (pp) and per-predictor contributions (pp).
            approval_contrib + gdp_contrib + unemp_contrib + cpi_contrib + intercept = total.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self.is_fitted_:
            raise RuntimeError("FundamentalsModel.predict() called before fit()")

        x = np.array([approval_net, gdp_q2_growth, unemployment, cpi_yoy])
        contribs = self.coef_ * x  # element-wise
        total = float(contribs.sum() + self.intercept_)
        return (
            total,
            float(contribs[0]),
            float(contribs[1]),
            float(contribs[2]),
            float(contribs[3]),
        )

    @classmethod
    def from_default_data(
        cls,
        history_path: Path | str | None = None,
        alpha: float = _DEFAULT_RIDGE_ALPHA,
    ) -> "FundamentalsModel":
        """Convenience constructor: load default CSV and fit in one call.

        Parameters
        ----------
        history_path:
            Path to history CSV.  Defaults to project default.
        alpha:
            Ridge regularization strength.

        Returns
        -------
        FundamentalsModel
            Fitted model.
        """
        model = cls(alpha=alpha)
        records = load_historical_data(history_path)
        model.fit(records)
        return model


# ---------------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------------

_SNAPSHOT_REQUIRED_KEYS = {
    "cycle", "in_party", "approval_net_oct",
    "gdp_q2_growth_pct", "unemployment_oct", "cpi_yoy_oct",
}


def _validate_snapshot_json(raw: dict, snapshot_path: Path) -> str:
    """Validate required keys and in_party value; return normalised in_party.

    Raises
    ------
    ValueError
        If required keys are missing or in_party is not "D" or "R".
    """
    missing = _SNAPSHOT_REQUIRED_KEYS - set(raw.keys())
    if missing:
        raise ValueError(f"snapshot JSON missing required keys: {missing}")

    in_party = raw["in_party"].strip().upper()
    if in_party not in {"D", "R"}:
        raise ValueError(f"in_party must be 'D' or 'R', got {in_party!r}")

    return in_party


def _parse_snapshot_json(raw: dict, in_party: str) -> FundamentalsSnapshot:
    """Build a FundamentalsSnapshot from a validated raw JSON dict."""
    return FundamentalsSnapshot(
        cycle=int(raw["cycle"]),
        in_party=in_party,
        approval_net_oct=float(raw["approval_net_oct"]),
        gdp_q2_growth_pct=float(raw["gdp_q2_growth_pct"]),
        unemployment_oct=float(raw["unemployment_oct"]),
        cpi_yoy_oct=float(raw["cpi_yoy_oct"]),
        consumer_sentiment=raw.get("consumer_sentiment"),
        source_notes=raw.get("source_notes", {}),
    )


def load_fundamentals_snapshot(
    snapshot_path: Path | str | None = None,
) -> FundamentalsSnapshot:
    """Load the current-cycle fundamentals snapshot from JSON.

    Parameters
    ----------
    snapshot_path:
        Path to the snapshot JSON file.  Defaults to
        ``data/fundamentals/snapshot_2026.json`` relative to project root.

    Returns
    -------
    FundamentalsSnapshot
        Parsed snapshot with current-cycle inputs.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    ValueError
        If required keys are missing from the JSON.
    """
    if snapshot_path is None:
        snapshot_path = _DEFAULT_SNAPSHOT_PATH
    snapshot_path = Path(snapshot_path)

    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Fundamentals snapshot not found at {snapshot_path}. "
            "Create data/fundamentals/snapshot_2026.json or pass an explicit path."
        )

    with snapshot_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    in_party = _validate_snapshot_json(raw, snapshot_path)
    snapshot = _parse_snapshot_json(raw, in_party)

    log.debug(
        "Loaded fundamentals snapshot: cycle=%d in_party=%s approval=%.1f gdp=%.1f unemp=%.1f cpi=%.1f",
        snapshot.cycle, snapshot.in_party,
        snapshot.approval_net_oct, snapshot.gdp_q2_growth_pct,
        snapshot.unemployment_oct, snapshot.cpi_yoy_oct,
    )
    return snapshot


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------


def _build_fundamentals_info(
    model: FundamentalsModel,
    total_pp: float,
    approval_pp: float,
    gdp_pp: float,
    unemp_pp: float,
    cpi_pp: float,
) -> FundamentalsInfo:
    """Convert per-component pp outputs from the Ridge model into a FundamentalsInfo.

    All Ridge outputs are in percentage-point units (Dem House share change).
    We multiply by _PP_TO_FRACTION to produce Dem-share fractions compatible
    with the county prior pipeline and generic ballot shift machinery.
    """
    return FundamentalsInfo(
        shift=total_pp * _PP_TO_FRACTION,
        approval_contribution=approval_pp * _PP_TO_FRACTION,
        gdp_contribution=gdp_pp * _PP_TO_FRACTION,
        unemployment_contribution=unemp_pp * _PP_TO_FRACTION,
        cpi_contribution=cpi_pp * _PP_TO_FRACTION,
        intercept_contribution=model.intercept_ * _PP_TO_FRACTION,
        loo_rmse=model.loo_rmse_ * _PP_TO_FRACTION,
        n_training=model.n_training_,
        source="fitted_ridge",
    )


def compute_fundamentals_shift(
    snapshot: FundamentalsSnapshot,
    history_path: Path | str | None = None,
    alpha: float = _DEFAULT_RIDGE_ALPHA,
    _model: Optional[FundamentalsModel] = None,
) -> FundamentalsInfo:
    """Compute the national fundamentals shift for the current cycle.

    Fits a Ridge regression on historical midterm data, then predicts the
    expected Dem House share change given current-cycle inputs.  The result
    ``shift`` is in the same units as the generic ballot shift — it is a
    fraction (not pp) relative to the 2024 presidential Dem share baseline.

    Note: The Ridge model predicts in **percentage points** of Dem House share
    change (i.e., the same units as ``dem_house_share_change_pp`` in the CSV).
    We convert to a Dem-share fraction by dividing by 100 before returning, so
    ``shift`` can be passed directly to ``apply_fundamentals_shift()`` or
    the generic ballot machinery.

    Parameters
    ----------
    snapshot:
        Current-cycle inputs from ``load_fundamentals_snapshot()``.
    history_path:
        Path to historical CSV.  Defaults to project default.
    alpha:
        Ridge regularization strength.  Override for sensitivity analysis.
    _model:
        Pre-fitted model.  If provided, skips fitting (used in tests for speed).

    Returns
    -------
    FundamentalsInfo
        Predicted shift and per-component contributions, plus LOO uncertainty.
    """
    if _model is None:
        _model = FundamentalsModel.from_default_data(history_path, alpha=alpha)

    total_pp, approval_pp, gdp_pp, unemp_pp, cpi_pp = _model.predict(
        approval_net=snapshot.approval_net_oct,
        gdp_q2_growth=snapshot.gdp_q2_growth_pct,
        unemployment=snapshot.unemployment_oct,
        cpi_yoy=snapshot.cpi_yoy_oct,
    )
    info = _build_fundamentals_info(_model, total_pp, approval_pp, gdp_pp, unemp_pp, cpi_pp)

    log.info(
        "Fundamentals shift: %.4f (%.2f pp) | approval=%.4f gdp=%.4f unemp=%.4f cpi=%.4f | LOO RMSE=%.4f",
        info.shift, total_pp,
        info.approval_contribution, info.gdp_contribution,
        info.unemployment_contribution, info.cpi_contribution,
        info.loo_rmse,
    )
    return info


# ---------------------------------------------------------------------------
# Apply shift to county priors
# ---------------------------------------------------------------------------


def apply_fundamentals_shift(
    county_priors: "np.ndarray",
    shift: float,
) -> "np.ndarray":
    """Apply a flat national fundamentals shift to county priors.

    Architecturally identical to ``apply_gb_shift()`` in ``generic_ballot.py``.
    Each county's prior is shifted by the same amount (``shift``), then clipped
    to [_PRIOR_MIN, _PRIOR_MAX] to keep values in a valid probability range.

    Parameters
    ----------
    county_priors:
        ndarray of shape (N,), county-level prior Dem shares.
    shift:
        Flat shift in Dem share units (e.g. +0.016 for +1.6pp D improvement).
        Use ``FundamentalsInfo.shift`` from ``compute_fundamentals_shift()``.

    Returns
    -------
    ndarray of shape (N,)
        Adjusted county priors, clipped to [0.01, 0.99].
    """
    adjusted = county_priors.astype(float) + shift
    return np.clip(adjusted, _PRIOR_MIN, _PRIOR_MAX)
