"""Core implementation of consistency-calibrated relative LLM views.

The module is deliberately independent from any LLM provider.  It accepts
pairwise predictions, calibrates them using *past* outcomes, projects them to
a globally consistent latent ranking, and maps them to Black--Litterman views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar


PROBABILITY_EPSILON = 1e-6


def _clip_probability(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"probability must be finite and in [0, 1], got {value}")
    return float(np.clip(value, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON))


def _as_string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    return tuple(str(item) for item in value if str(item).strip())


def canonical_pair(asset_a: str, asset_b: str) -> tuple[str, str]:
    if asset_a == asset_b:
        raise ValueError("a relative view must contain two different assets")
    return tuple(sorted((str(asset_a), str(asset_b))))  # type: ignore[return-value]


@dataclass(frozen=True)
class PairwiseView:
    """Probability that ``asset_a`` outperforms ``asset_b`` over the horizon."""

    asset_a: str
    asset_b: str
    probability_a: float
    probability_samples_a: tuple[float, ...] = ()
    evidence: tuple[str, ...] = ()
    weight: float = 1.0
    horizon_days: int | None = None

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "PairwiseView":
        asset_a = str(record["asset_a"])
        asset_b = str(record["asset_b"])
        canonical_pair(asset_a, asset_b)

        preferred = record.get("preferred_asset")
        if "probability_a" in record:
            probability_a = _clip_probability(record["probability_a"])
            samples = record.get("probability_samples_a", record.get("probability_samples", ()))
            samples_a = tuple(_clip_probability(item) for item in samples)
        else:
            probability = _clip_probability(record["probability"])
            if preferred not in (asset_a, asset_b):
                raise ValueError("preferred_asset must equal asset_a or asset_b")
            probability_a = probability if preferred == asset_a else 1.0 - probability
            raw_samples = record.get("probability_samples", record.get("repeated_probabilities", ()))
            samples_a = tuple(
                _clip_probability(item) if preferred == asset_a else 1.0 - _clip_probability(item)
                for item in raw_samples
            )

        weight = float(record.get("weight", 1.0))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("view weight must be finite and positive")
        horizon = record.get("horizon_days")
        return cls(
            asset_a=asset_a,
            asset_b=asset_b,
            probability_a=probability_a,
            probability_samples_a=samples_a,
            evidence=_as_string_list(record.get("evidence")),
            weight=weight,
            horizon_days=int(horizon) if horizon is not None else None,
        )

    def canonicalized(self) -> "PairwiseView":
        left, right = canonical_pair(self.asset_a, self.asset_b)
        if (left, right) == (self.asset_a, self.asset_b):
            return self
        return PairwiseView(
            asset_a=left,
            asset_b=right,
            probability_a=1.0 - self.probability_a,
            probability_samples_a=tuple(1.0 - item for item in self.probability_samples_a),
            evidence=self.evidence,
            weight=self.weight,
            horizon_days=self.horizon_days,
        )


@dataclass(frozen=True)
class CalibrationObservation:
    asset_a: str
    asset_b: str
    probability_a: float
    outcome_a: int

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "CalibrationObservation":
        asset_a = str(record["asset_a"])
        asset_b = str(record["asset_b"])
        probability_a = _clip_probability(record.get("probability_a", record.get("probability")))
        if "outcome_a" in record:
            outcome = int(record["outcome_a"])
        elif "outcome" in record:
            outcome = int(record["outcome"])
        elif "realized_winner" in record:
            winner = str(record["realized_winner"])
            if winner not in (asset_a, asset_b):
                raise ValueError("realized_winner must equal asset_a or asset_b")
            outcome = int(winner == asset_a)
        else:
            raise ValueError("calibration observation is missing an outcome")
        if outcome not in (0, 1):
            raise ValueError("calibration outcome must be 0 or 1")
        return cls(asset_a, asset_b, probability_a, outcome)

    def canonicalized(self) -> "CalibrationObservation":
        left, right = canonical_pair(self.asset_a, self.asset_b)
        if (left, right) == (self.asset_a, self.asset_b):
            return self
        return CalibrationObservation(left, right, 1.0 - self.probability_a, 1 - self.outcome_a)


class ProbabilityCalibrator:
    """Identity, temperature, or dependency-free isotonic calibration."""

    def __init__(self, method: str = "isotonic", min_samples: int = 20):
        if method not in {"none", "temperature", "isotonic"}:
            raise ValueError("calibration method must be none, temperature, or isotonic")
        self.method = method
        self.min_samples = int(min_samples)
        self.fitted_method = "none"
        self.temperature = 1.0
        self.thresholds = np.array([], dtype=float)
        self.values = np.array([], dtype=float)
        # With no history, use the uninformative Bernoulli Brier score rather
        # than treating missing calibration evidence as perfect calibration.
        self.training_brier = 0.25

    def fit(self, probabilities: Sequence[float], outcomes: Sequence[int]) -> "ProbabilityCalibrator":
        probabilities_array = np.asarray(probabilities, dtype=float)
        outcomes_array = np.asarray(outcomes, dtype=float)
        if probabilities_array.shape != outcomes_array.shape:
            raise ValueError("probabilities and outcomes must have the same shape")
        if probabilities_array.ndim != 1:
            raise ValueError("calibration inputs must be one-dimensional")
        if not np.all(np.isfinite(probabilities_array)) or np.any((probabilities_array < 0) | (probabilities_array > 1)):
            raise ValueError("calibration probabilities must be finite and in [0, 1]")
        if np.any((outcomes_array != 0) & (outcomes_array != 1)):
            raise ValueError("calibration outcomes must be binary")
        if len(probabilities_array) < self.min_samples or self.method == "none":
            self.fitted_method = "none"
            if len(probabilities_array):
                self.training_brier = float(np.mean((probabilities_array - outcomes_array) ** 2))
            return self
        probabilities_array = np.clip(probabilities_array, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)

        if self.method == "temperature":
            logits = np.log(probabilities_array / (1.0 - probabilities_array))

            def negative_log_likelihood(log_temperature: float) -> float:
                temperature = np.exp(log_temperature)
                predictions = 1.0 / (1.0 + np.exp(-logits / temperature))
                predictions = np.clip(predictions, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
                return float(-np.mean(
                    outcomes_array * np.log(predictions)
                    + (1.0 - outcomes_array) * np.log(1.0 - predictions)
                ))

            result = minimize_scalar(
                negative_log_likelihood,
                bounds=(np.log(0.05), np.log(20.0)),
                method="bounded",
            )
            self.temperature = float(np.exp(result.x))
            self.fitted_method = "temperature"
        else:
            self._fit_isotonic(probabilities_array, outcomes_array)
            self.fitted_method = "isotonic"

        predictions = self.predict(probabilities_array)
        self.training_brier = float(np.mean((predictions - outcomes_array) ** 2))
        return self

    def _fit_isotonic(self, probabilities: np.ndarray, outcomes: np.ndarray) -> None:
        order = np.argsort(probabilities, kind="stable")
        x_sorted = probabilities[order]
        y_sorted = outcomes[order]
        unique_x, inverse, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
        sums = np.bincount(inverse, weights=y_sorted)

        blocks: list[dict[str, float]] = []
        for x_value, count, total in zip(unique_x, counts, sums):
            blocks.append({"upper": float(x_value), "weight": float(count), "mean": float(total / count)})
            while len(blocks) >= 2 and blocks[-2]["mean"] > blocks[-1]["mean"]:
                right = blocks.pop()
                left = blocks.pop()
                weight = left["weight"] + right["weight"]
                blocks.append({
                    "upper": right["upper"],
                    "weight": weight,
                    "mean": (left["weight"] * left["mean"] + right["weight"] * right["mean"]) / weight,
                })
        self.thresholds = np.asarray([block["upper"] for block in blocks], dtype=float)
        self.values = np.asarray([block["mean"] for block in blocks], dtype=float)

    def predict(self, probabilities: float | Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float)
        clipped = np.clip(values, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
        if self.fitted_method == "temperature":
            logits = np.log(clipped / (1.0 - clipped))
            result = 1.0 / (1.0 + np.exp(-logits / self.temperature))
        elif self.fitted_method == "isotonic":
            indices = np.searchsorted(self.thresholds, clipped, side="left")
            indices = np.clip(indices, 0, len(self.values) - 1)
            result = self.values[indices]
        else:
            result = clipped
        return np.clip(result, PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)


def expected_calibration_error(
    probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> float:
    probabilities_array = np.asarray(probabilities, dtype=float)
    outcomes_array = np.asarray(outcomes, dtype=float)
    if not len(probabilities_array):
        return 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities_array >= boundaries[index]) & (probabilities_array <= boundaries[index + 1])
        else:
            mask = (probabilities_array >= boundaries[index]) & (probabilities_array < boundaries[index + 1])
        if np.any(mask):
            total += float(np.mean(mask)) * abs(float(np.mean(probabilities_array[mask]) - np.mean(outcomes_array[mask])))
    return total


def calibration_diagnostics(
    observations: Iterable[CalibrationObservation | Mapping[str, Any]],
    calibrator: ProbabilityCalibrator,
    bins: int = 10,
) -> dict[str, Any]:
    parsed = [
        (item if isinstance(item, CalibrationObservation) else CalibrationObservation.from_mapping(item)).canonicalized()
        for item in observations
    ]
    if not parsed:
        return {"count": 0, "brier": 0.0, "ece": 0.0, "directional_accuracy": 0.0, "reliability": []}
    outcomes = np.asarray([item.outcome_a for item in parsed], dtype=float)
    calibrated = calibrator.predict([item.probability_a for item in parsed])
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    reliability: list[dict[str, Any]] = []
    for index in range(bins):
        right_closed = index == bins - 1
        mask = (calibrated >= boundaries[index]) & (
            (calibrated <= boundaries[index + 1]) if right_closed else (calibrated < boundaries[index + 1])
        )
        if np.any(mask):
            reliability.append({
                "lower": float(boundaries[index]),
                "upper": float(boundaries[index + 1]),
                "count": int(np.sum(mask)),
                "mean_probability": float(np.mean(calibrated[mask])),
                "outcome_rate": float(np.mean(outcomes[mask])),
            })
    return {
        "count": len(parsed),
        "brier": float(np.mean((calibrated - outcomes) ** 2)),
        "ece": expected_calibration_error(calibrated, outcomes, bins),
        "directional_accuracy": float(np.mean((calibrated >= 0.5) == outcomes.astype(bool))),
        "reliability": reliability,
    }


@dataclass(frozen=True)
class RelViewConfig:
    calibration: str = "isotonic"
    min_calibration_samples: int = 20
    abstention_threshold: float = 0.60
    min_evidence: int = 1
    consistency_lambda: float = 1e-3
    entropy_weight: float = 0.5
    disagreement_weight: float = 0.3
    calibration_error_weight: float = 0.2
    omega_epsilon: float = 1e-8
    tau: float = 0.025
    risk_aversion: float = 0.1
    turnover_penalty: float = 0.0
    max_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.calibration not in {"none", "temperature", "isotonic"}:
            raise ValueError("calibration must be none, temperature, or isotonic")
        if self.min_calibration_samples < 0:
            raise ValueError("min_calibration_samples cannot be negative")
        if not 0.5 <= self.abstention_threshold <= 1.0:
            raise ValueError("abstention_threshold must be in [0.5, 1]")
        if self.min_evidence < 0:
            raise ValueError("min_evidence cannot be negative")
        if self.consistency_lambda < 0 or self.omega_epsilon <= 0 or self.tau <= 0:
            raise ValueError("consistency_lambda, omega_epsilon, and tau must be valid positive scales")
        uncertainty_weights = (self.entropy_weight, self.disagreement_weight, self.calibration_error_weight)
        if any(weight < 0 for weight in uncertainty_weights) or sum(uncertainty_weights) <= 0:
            raise ValueError("uncertainty weights must be non-negative and have a positive sum")
        if self.risk_aversion < 0 or self.turnover_penalty < 0:
            raise ValueError("risk_aversion and turnover_penalty must be non-negative")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")


@dataclass
class RelViewMatrices:
    P: np.ndarray
    q: np.ndarray
    omega: np.ndarray
    latent_scores: pd.Series
    accepted_views: list[dict[str, Any]] = field(default_factory=list)
    rejected_views: list[dict[str, Any]] = field(default_factory=list)
    raw_cycle_count: int = 0
    consistency_rmse: float = 0.0


@dataclass
class RelViewResult:
    assets: list[str]
    weights: pd.Series
    posterior_returns: pd.Series
    matrices: RelViewMatrices
    calibrator: ProbabilityCalibrator
    optimizer_message: str


def aggregate_pairwise_views(views: Iterable[PairwiseView | Mapping[str, Any]]) -> list[PairwiseView]:
    grouped: dict[tuple[str, str], list[PairwiseView]] = {}
    for item in views:
        view = item if isinstance(item, PairwiseView) else PairwiseView.from_mapping(item)
        view = view.canonicalized()
        grouped.setdefault((view.asset_a, view.asset_b), []).append(view)

    aggregated: list[PairwiseView] = []
    for (asset_a, asset_b), group in grouped.items():
        weights = np.asarray([item.weight for item in group], dtype=float)
        probability = float(np.average([item.probability_a for item in group], weights=weights))
        samples = tuple(sample for item in group for sample in item.probability_samples_a)
        if not samples and len(group) > 1:
            samples = tuple(item.probability_a for item in group)
        evidence = tuple(dict.fromkeys(piece for item in group for piece in item.evidence))
        horizons = [item.horizon_days for item in group if item.horizon_days is not None]
        aggregated.append(PairwiseView(
            asset_a=asset_a,
            asset_b=asset_b,
            probability_a=probability,
            probability_samples_a=samples,
            evidence=evidence,
            weight=float(np.sum(weights)),
            horizon_days=int(round(np.mean(horizons))) if horizons else None,
        ))
    return aggregated


def fit_calibrator(
    observations: Iterable[CalibrationObservation | Mapping[str, Any]],
    method: str = "isotonic",
    min_samples: int = 20,
) -> tuple[ProbabilityCalibrator, dict[tuple[str, str], float]]:
    parsed = [
        item.canonicalized() if isinstance(item, CalibrationObservation)
        else CalibrationObservation.from_mapping(item).canonicalized()
        for item in observations
    ]
    calibrator = ProbabilityCalibrator(method, min_samples)
    calibrator.fit([item.probability_a for item in parsed], [item.outcome_a for item in parsed])

    errors: dict[tuple[str, str], list[float]] = {}
    if parsed:
        calibrated = calibrator.predict([item.probability_a for item in parsed])
        for item, probability in zip(parsed, calibrated):
            errors.setdefault((item.asset_a, item.asset_b), []).append((float(probability) - item.outcome_a) ** 2)
    pair_errors = {key: float(np.mean(values)) for key, values in errors.items()}
    return calibrator, pair_errors


def _normalized_entropy(probability: float) -> float:
    p = _clip_probability(probability)
    return float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0))


def project_consistent_scores(
    assets: Sequence[str],
    comparisons: Sequence[tuple[str, str, float, float]],
    ridge: float = 1e-3,
) -> pd.Series:
    """Fit theta_i-theta_j to calibrated pairwise logits by weighted ridge LS."""

    asset_list = list(assets)
    index = {asset: position for position, asset in enumerate(asset_list)}
    if not comparisons:
        return pd.Series(np.zeros(len(asset_list)), index=asset_list, dtype=float)
    design = np.zeros((len(comparisons), len(asset_list)), dtype=float)
    target = np.zeros(len(comparisons), dtype=float)
    weights = np.zeros(len(comparisons), dtype=float)
    for row, (asset_a, asset_b, probability_a, weight) in enumerate(comparisons):
        design[row, index[asset_a]] = 1.0
        design[row, index[asset_b]] = -1.0
        probability_a = _clip_probability(probability_a)
        target[row] = np.log(probability_a / (1.0 - probability_a))
        weights[row] = max(float(weight), PROBABILITY_EPSILON)
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_target = target * np.sqrt(weights)
    lhs = weighted_design.T @ weighted_design + ridge * np.eye(len(asset_list))
    rhs = weighted_design.T @ weighted_target
    scores = np.linalg.pinv(lhs) @ rhs
    scores -= np.mean(scores)
    return pd.Series(scores, index=asset_list, dtype=float)


def count_preference_cycles(assets: Sequence[str], views: Sequence[tuple[str, str, float]]) -> int:
    index = {asset: position for position, asset in enumerate(assets)}
    adjacency = np.zeros((len(assets), len(assets)), dtype=np.int64)
    for asset_a, asset_b, probability_a in views:
        winner, loser = (asset_a, asset_b) if probability_a > 0.5 else (asset_b, asset_a)
        adjacency[index[winner], index[loser]] = 1
    return int(np.trace(adjacency @ adjacency @ adjacency) // 3)


def cross_sectional_return_scale(returns: pd.DataFrame | np.ndarray) -> float:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("returns must contain at least two assets")
    scale = float(np.nanmean(np.nanstd(values, axis=1, ddof=1)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanmean(np.nanstd(values, axis=0, ddof=1)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("cannot estimate a positive cross-sectional return scale")
    return scale


def build_relview_matrices(
    assets: Sequence[str],
    views: Iterable[PairwiseView | Mapping[str, Any]],
    return_scale: float,
    calibrator: ProbabilityCalibrator | None = None,
    pair_calibration_errors: Mapping[tuple[str, str], float] | None = None,
    config: RelViewConfig | None = None,
) -> RelViewMatrices:
    config = config or RelViewConfig()
    calibrator = calibrator or ProbabilityCalibrator("none", 0).fit([], [])
    pair_calibration_errors = pair_calibration_errors or {}
    asset_list = list(assets)
    asset_set = set(asset_list)
    index = {asset: position for position, asset in enumerate(asset_list)}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for view in aggregate_pairwise_views(views):
        base = {"asset_a": view.asset_a, "asset_b": view.asset_b, "raw_probability_a": view.probability_a}
        if view.asset_a not in asset_set or view.asset_b not in asset_set:
            rejected.append({**base, "reason": "asset_not_in_universe"})
            continue
        probability = float(calibrator.predict([view.probability_a])[0])
        if max(probability, 1.0 - probability) < config.abstention_threshold:
            rejected.append({**base, "calibrated_probability_a": probability, "reason": "low_confidence"})
            continue
        if len(view.evidence) < config.min_evidence:
            rejected.append({**base, "calibrated_probability_a": probability, "reason": "insufficient_evidence"})
            continue

        calibrated_samples = calibrator.predict(view.probability_samples_a) if view.probability_samples_a else np.array([])
        disagreement = float(np.std(calibrated_samples, ddof=1)) if len(calibrated_samples) > 1 else 0.0
        disagreement = min(disagreement / 0.5, 1.0)
        calibration_error = float(np.clip(
            pair_calibration_errors.get((view.asset_a, view.asset_b), calibrator.training_brier), 0.0, 1.0
        ))
        numerator = (
            config.entropy_weight * _normalized_entropy(probability)
            + config.disagreement_weight * disagreement
            + config.calibration_error_weight * calibration_error
        )
        denominator = config.entropy_weight + config.disagreement_weight + config.calibration_error_weight
        uncertainty = float(np.clip(numerator / denominator, PROBABILITY_EPSILON, 1.0))
        accepted.append({
            **base,
            "calibrated_probability_a": probability,
            "uncertainty": uncertainty,
            "disagreement": disagreement,
            "calibration_error": calibration_error,
            "weight": view.weight,
            "evidence": list(view.evidence),
        })

    comparisons = [
        (
            item["asset_a"],
            item["asset_b"],
            item["calibrated_probability_a"],
            item["weight"] / max(item["uncertainty"], PROBABILITY_EPSILON),
        )
        for item in accepted
    ]
    scores = project_consistent_scores(asset_list, comparisons, config.consistency_lambda)
    P = np.zeros((len(accepted), len(asset_list)), dtype=float)
    q = np.zeros(len(accepted), dtype=float)
    omega_diagonal = np.zeros(len(accepted), dtype=float)
    for row, item in enumerate(accepted):
        asset_a = item["asset_a"]
        asset_b = item["asset_b"]
        P[row, index[asset_a]] = 1.0
        P[row, index[asset_b]] = -1.0
        score_difference = float(scores.loc[asset_a] - scores.loc[asset_b])
        q[row] = return_scale * np.tanh(score_difference)
        omega_diagonal[row] = (return_scale * item["uncertainty"]) ** 2 + config.omega_epsilon
        item["projected_score_difference"] = score_difference
        raw_logit = float(np.log(
            item["calibrated_probability_a"] / (1.0 - item["calibrated_probability_a"])
        ))
        item["consistency_residual"] = raw_logit - score_difference
        item["q"] = float(q[row])
        item["omega"] = float(omega_diagonal[row])

    cycle_views = [
        (item["asset_a"], item["asset_b"], item["calibrated_probability_a"])
        for item in accepted
    ]
    consistency_weights = np.asarray([item[3] for item in comparisons], dtype=float)
    consistency_residuals = np.asarray(
        [item["consistency_residual"] for item in accepted], dtype=float
    )
    consistency_rmse = (
        float(np.sqrt(np.average(consistency_residuals ** 2, weights=consistency_weights)))
        if len(consistency_residuals) else 0.0
    )
    return RelViewMatrices(
        P=P,
        q=q,
        omega=np.diag(omega_diagonal),
        latent_scores=scores,
        accepted_views=accepted,
        rejected_views=rejected,
        raw_cycle_count=count_preference_cycles(asset_list, cycle_views),
        consistency_rmse=consistency_rmse,
    )


def black_litterman_posterior(
    prior_returns: Sequence[float] | np.ndarray,
    covariance: np.ndarray,
    P: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
    tau: float = 0.025,
) -> np.ndarray:
    prior = np.asarray(prior_returns, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (len(prior), len(prior)):
        raise ValueError("covariance shape must match prior_returns")
    if P.shape[1] != len(prior) or len(q) != P.shape[0] or omega.shape != (len(q), len(q)):
        raise ValueError("P, q, and omega have incompatible shapes")
    if len(q) == 0:
        return prior.copy()
    inverse_tau_covariance = np.linalg.pinv(tau * covariance)
    inverse_omega = np.linalg.pinv(omega)
    middle = np.linalg.pinv(inverse_tau_covariance + P.T @ inverse_omega @ P)
    return middle @ (inverse_tau_covariance @ prior + P.T @ inverse_omega @ q)


def optimize_portfolio(
    expected_returns: Sequence[float] | np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float = 0.1,
    turnover_penalty: float = 0.0,
    previous_weights: Sequence[float] | np.ndarray | None = None,
    max_weight: float = 0.1,
) -> tuple[np.ndarray, str]:
    expected = np.asarray(expected_returns, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    asset_count = len(expected)
    if max_weight * asset_count < 1.0 - 1e-12:
        raise ValueError(f"max_weight={max_weight} is infeasible for {asset_count} assets")
    previous = np.asarray(previous_weights, dtype=float) if previous_weights is not None else None
    if previous is not None and previous.shape != expected.shape:
        raise ValueError("previous_weights must match expected_returns")

    def objective(weights: np.ndarray) -> float:
        utility = weights @ expected - risk_aversion * (weights.T @ covariance @ weights)
        turnover = turnover_penalty * np.sum(np.abs(weights - previous)) if previous is not None else 0.0
        return float(-utility + turnover)

    initial = previous.copy() if previous is not None else np.full(asset_count, 1.0 / asset_count)
    initial = np.clip(initial, 0.0, max_weight)
    if initial.sum() <= 0:
        initial = np.full(asset_count, 1.0 / asset_count)
    else:
        initial /= initial.sum()
    if np.max(initial) > max_weight + 1e-12:
        initial = np.full(asset_count, 1.0 / asset_count)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * asset_count,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"portfolio optimization failed: {result.message}")
    return np.asarray(result.x, dtype=float), str(result.message)


def implied_equilibrium_returns(
    covariance: np.ndarray,
    market_weights: Sequence[float] | np.ndarray,
    market_risk_aversion: float = 2.5,
) -> np.ndarray:
    weights = np.asarray(market_weights, dtype=float)
    weights = weights / weights.sum()
    return float(market_risk_aversion) * np.asarray(covariance, dtype=float) @ weights


def run_relview_bl(
    returns: pd.DataFrame,
    prior_returns: Sequence[float] | Mapping[str, float] | pd.Series,
    views: Iterable[PairwiseView | Mapping[str, Any]],
    calibration_history: Iterable[CalibrationObservation | Mapping[str, Any]] = (),
    config: RelViewConfig | None = None,
    previous_weights: Sequence[float] | Mapping[str, float] | pd.Series | None = None,
) -> RelViewResult:
    config = config or RelViewConfig()
    clean_returns = (
        returns.select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .dropna(axis=1, how="any")
    )
    assets = clean_returns.columns.astype(str).tolist()
    if len(assets) < 2:
        raise ValueError("at least two complete asset return series are required")
    covariance = clean_returns.cov().to_numpy(dtype=float)

    def align(values: Sequence[float] | Mapping[str, float] | pd.Series, name: str) -> np.ndarray:
        if isinstance(values, Mapping) or isinstance(values, pd.Series):
            series = pd.Series(values, dtype=float)
            missing = [asset for asset in assets if asset not in series.index]
            if missing:
                raise ValueError(f"{name} is missing assets: {missing}")
            return series.loc[assets].to_numpy(dtype=float)
        array = np.asarray(values, dtype=float)
        if array.shape != (len(assets),):
            raise ValueError(f"{name} must contain {len(assets)} values")
        return array

    prior = align(prior_returns, "prior_returns")
    previous = align(previous_weights, "previous_weights") if previous_weights is not None else None
    calibrator, pair_errors = fit_calibrator(
        calibration_history, config.calibration, config.min_calibration_samples
    )
    matrices = build_relview_matrices(
        assets,
        views,
        cross_sectional_return_scale(clean_returns),
        calibrator,
        pair_errors,
        config,
    )
    posterior = black_litterman_posterior(
        prior, covariance, matrices.P, matrices.q, matrices.omega, config.tau
    )
    weights, message = optimize_portfolio(
        posterior,
        covariance,
        config.risk_aversion,
        config.turnover_penalty,
        previous,
        config.max_weight,
    )
    return RelViewResult(
        assets=assets,
        weights=pd.Series(weights, index=assets, name="weight"),
        posterior_returns=pd.Series(posterior, index=assets, name="posterior_return"),
        matrices=matrices,
        calibrator=calibrator,
        optimizer_message=message,
    )


def select_candidate_pairs(
    returns: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    market_caps: Mapping[str, float] | None = None,
    max_pairs: int = 50,
    min_abs_correlation: float = 0.0,
) -> list[tuple[str, str]]:
    """Select a sparse comparison graph using correlation, sector, and size."""

    assets = returns.select_dtypes(include=[np.number]).columns.astype(str).tolist()
    correlations = returns[assets].corr()
    sector_by_asset: dict[str, str] = {}
    if metadata is not None:
        symbol_column = next((item for item in ("Symbol", "symbol", "ticker") if item in metadata.columns), None)
        sector_column = next((item for item in ("GICS Sector", "sector", "Sector") if item in metadata.columns), None)
        if symbol_column and sector_column:
            sector_by_asset = dict(zip(metadata[symbol_column].astype(str), metadata[sector_column].astype(str)))

    candidates: list[tuple[float, str, str]] = []
    for asset_a, asset_b in combinations(assets, 2):
        correlation = abs(float(correlations.loc[asset_a, asset_b]))
        if not np.isfinite(correlation) or correlation < min_abs_correlation:
            continue
        same_sector = float(
            asset_a in sector_by_asset
            and asset_b in sector_by_asset
            and sector_by_asset[asset_a] == sector_by_asset[asset_b]
        )
        size_similarity = 0.0
        if market_caps and asset_a in market_caps and asset_b in market_caps:
            cap_a, cap_b = float(market_caps[asset_a]), float(market_caps[asset_b])
            if cap_a > 0 and cap_b > 0:
                size_similarity = float(np.exp(-abs(np.log(cap_a / cap_b))))
        score = correlation + 0.5 * same_sector + 0.25 * size_similarity
        candidates.append((score, asset_a, asset_b))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    if max_pairs < max(len(assets) - 1, 1):
        return [(asset_a, asset_b) for _, asset_a, asset_b in candidates[:max_pairs]]

    # A maximum-score spanning forest prevents the consistency graph from
    # needlessly fragmenting when the requested pair budget permits coverage.
    parent = {asset: asset for asset in assets}

    def find(asset: str) -> str:
        while parent[asset] != asset:
            parent[asset] = parent[parent[asset]]
            asset = parent[asset]
        return asset

    selected: list[tuple[str, str]] = []
    selected_set: set[tuple[str, str]] = set()
    for _, asset_a, asset_b in candidates:
        root_a, root_b = find(asset_a), find(asset_b)
        if root_a != root_b:
            parent[root_b] = root_a
            pair = (asset_a, asset_b)
            selected.append(pair)
            selected_set.add(pair)
            if len(selected) == len(assets) - 1:
                break
    for _, asset_a, asset_b in candidates:
        pair = (asset_a, asset_b)
        if pair not in selected_set:
            selected.append(pair)
            selected_set.add(pair)
        if len(selected) >= max_pairs:
            break
    return selected[:max_pairs]


def calibration_observations_from_realized_returns(
    views: Iterable[PairwiseView | Mapping[str, Any]],
    realized_returns: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Create observations after a horizon closes; never feed current outcomes early."""

    compounded = (1.0 + realized_returns).prod(axis=0) - 1.0
    observations: list[dict[str, Any]] = []
    for view in aggregate_pairwise_views(views):
        if view.asset_a not in compounded.index or view.asset_b not in compounded.index:
            continue
        return_a = float(compounded.loc[view.asset_a])
        return_b = float(compounded.loc[view.asset_b])
        observations.append({
            "asset_a": view.asset_a,
            "asset_b": view.asset_b,
            "probability_a": view.probability_a,
            "outcome_a": int(return_a > return_b),
            "realized_return_a": return_a,
            "realized_return_b": return_b,
        })
    return observations
