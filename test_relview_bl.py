import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_fortnight_results import (
    manifest_tickers,
    relative_diagnostics,
    weighted_probability_summary,
)
from collect_absolute_views import make_absolute_prompt, parse_absolute_response
from collect_relative_views import (
    aggregate_repeated_predictions,
    completion_request_options,
    make_pairwise_prompt,
    parse_pairwise_response,
    provider_limit_from_errors,
)
from portfolio_backtest import evaluate_realized_portfolio
from prompt_ensemble import collect_repeated_calls, diversified_system_prompt, prompt_sha256
from relview_bl import (
    PairwiseView,
    ProbabilityCalibrator,
    RelViewConfig,
    black_litterman_posterior,
    build_relview_matrices,
    calibration_observations_from_realized_returns,
    optimize_portfolio,
    run_relview_bl,
    select_candidate_pairs,
)
from validate_fortnight_data import metrics_from_daily_returns


class RelViewTests(unittest.TestCase):
    def test_independent_validator_metrics_match_portfolio_evaluator(self):
        returns = np.asarray([0.01, -0.02, 0.005, 0.003], dtype=float)
        _, expected = evaluate_realized_portfolio(
            pd.DataFrame({"A": returns}), {"A": 1.0}
        )
        actual = metrics_from_daily_returns(returns)
        for key in (
            "trading_days", "cumulative_return", "annualized_return",
            "annualized_volatility", "sharpe", "max_drawdown",
        ):
            self.assertAlmostEqual(actual[key], expected[key], places=12)

    def test_manifest_tickers_supports_new_and_previous_schemas(self):
        manifest = {"datasets": [
            {"tickers": ["A", "B"]},
            {"assets": [{"ticker": "C"}, {"ticker": "D"}]},
        ]}
        self.assertEqual(manifest_tickers(manifest), ["A", "B", "C", "D"])

    def test_relative_diagnostics_preserve_call_and_aggregate_distinction(self):
        with tempfile.TemporaryDirectory() as temporary:
            response_root = Path(temporary) / "responses_relative"
            response_root.mkdir()
            (response_root / "test_01.json").write_text(json.dumps({
                "probability_semantics": "confidence-forced ranking score",
                "views": [
                    {
                        "status": "ok",
                        "probability": 0.60,
                        "reported_probability_samples_a": [0.55, 0.65],
                        "votes": {"A": 2, "B": 0},
                        "successful_repeats": 2,
                    },
                    {
                        "status": "ok",
                        "probability": 0.50,
                        "reported_probability_samples_a": [0.55, 0.45],
                        "votes": {"A": 1, "B": 1},
                        "successful_repeats": 2,
                    },
                ],
            }), encoding="utf-8")
            diagnostics = relative_diagnostics(Path(temporary), 0.60)
            self.assertEqual(int(diagnostics.loc[0, "Pair_Views"]), 2)
            self.assertEqual(int(diagnostics.loc[0, "Successful_Calls"]), 4)
            self.assertAlmostEqual(diagnostics.loc[0, "Aggregate_Confidence_Mean"], 0.55)
            self.assertEqual(int(diagnostics.loc[0, "Near_Half_Below_0_55"]), 1)
            self.assertEqual(int(diagnostics.loc[0, "Accepted_At_Threshold"]), 1)
            self.assertAlmostEqual(diagnostics.loc[0, "Per_Call_Confidence_Min"], 0.55)
            summary = weighted_probability_summary(
                diagnostics.assign(Dataset="synthetic")
            )
            self.assertAlmostEqual(summary.loc[0, "Accepted_Share"], 0.5)

    def test_provider_usage_limit_parses_reset_delay(self):
        error = provider_limit_from_errors([
            "RateLimitError 429 GoUsageLimitError: 5-hour usage limit reached. "
            "Resets in 1hr 42min."
        ])
        self.assertIsNotNone(error)
        self.assertEqual(error.retry_after_seconds, 6180)

    def test_paper_optimizer_uses_variance_minus_point_one_return(self):
        expected = np.array([0.2, 0.0])
        covariance = np.diag([1.0, 0.0])
        legacy, _ = optimize_portfolio(
            expected, covariance, risk_aversion=0.1, max_weight=1.0,
            objective_convention="legacy_utility",
        )
        paper, _ = optimize_portfolio(
            expected, covariance, risk_aversion=0.1, max_weight=1.0,
            objective_convention="paper_variance_minus_return",
        )
        self.assertGreater(legacy[0], 0.9)
        self.assertAlmostEqual(paper[0], 0.01, places=3)

    def test_prompt_ensemble_has_unique_auditable_system_prompts(self):
        prompts = [
            diversified_system_prompt("return JSON only", call_id, "relative:A:B")
            for call_id in range(30)
        ]
        self.assertEqual(len(set(prompts)), 30)
        self.assertEqual(len({prompt_sha256(prompt) for prompt in prompts}), 30)
        self.assertTrue(all("return JSON only" in prompt for prompt in prompts))

    def test_prompt_ensemble_retry_uses_a_new_call_id(self):
        attempted = []

        def call_once(call_id):
            attempted.append(call_id)
            if call_id == 0:
                raise ValueError("invalid JSON")
            return call_id

        calls, errors, attempts = collect_repeated_calls(call_once, 3, 1, 2)
        self.assertEqual(attempted, [0, 1, 2, 3])
        self.assertEqual([call_id for call_id, _ in calls], [1, 2, 3])
        self.assertEqual(attempts, 4)
        self.assertEqual(len(errors), 1)

    def test_black_litterman_without_views_returns_prior(self):
        prior = np.array([0.01, 0.02, -0.005])
        covariance = np.eye(3) * 0.04
        posterior = black_litterman_posterior(
            prior,
            covariance,
            np.empty((0, 3)),
            np.empty(0),
            np.empty((0, 0)),
        )
        np.testing.assert_allclose(posterior, prior)

    def test_absolute_response_uses_decimal_daily_return(self):
        self.assertAlmostEqual(parse_absolute_response('{"expected_return": 0.0025}'), 0.0025)
        with self.assertRaises(ValueError):
            parse_absolute_response('{"expected_return": 3.0}')

    def test_paper_prompt_scales_inputs_and_parses_percentage_output(self):
        system, user = make_absolute_prompt(
            "AAPL",
            [0.01, -0.002],
            {
                "Security": "Apple Inc.",
                "GICS Sector": "Information Technology",
                "GICS Sub-Industry": "Hardware",
            },
            {
                "reference_date": "2024-08-30",
                "sector_returns": [0.5, -0.1],
                "market_returns": [0.2, -0.2],
            },
            10,
            "paper_v2",
        )
        payload = json.loads(user)
        self.assertEqual(payload["Daily Returns"], [1.0, -0.2])
        self.assertIn("2024-08-30", system)
        self.assertAlmostEqual(
            parse_absolute_response('{"expected_return": 0.25}', percentage_units=True),
            0.0025,
        )

    def test_decisive_prompt_enforces_minimum_ranking_score(self):
        returns = pd.DataFrame({"A": [0.01, 0.02], "B": [-0.01, 0.0]})
        context = {
            "A": {"sector_returns": [1.0, 2.0], "market_returns": [0.5, 0.6]},
            "B": {"sector_returns": [1.0, 2.0], "market_returns": [0.5, 0.6]},
        }
        system, user = make_pairwise_prompt(
            "A", "B", returns, {}, context, 10, "decisive_v3"
        )
        self.assertIn("[0.55, 0.95]", system)
        self.assertIn("fixed decision rule", system)
        with self.assertRaises(ValueError):
            parse_pairwise_response(
                '{"preferred_asset":"A","probability":0.96,"evidence":[]}',
                "A", "B", 0.55, 0.95,
            )
        self.assertEqual(json.loads(user)["asset_a"]["daily_returns"], [1.0, 2.0])
        with self.assertRaises(ValueError):
            parse_pairwise_response(
                '{"preferred_asset":"A","probability":0.54,"evidence":[]}',
                "A", "B", 0.55,
            )

    def test_realized_portfolio_metrics_and_transaction_cost(self):
        realized = pd.DataFrame({"A": [0.01, 0.02], "B": [0.0, -0.01]}, index=["d1", "d2"])
        daily, metrics = evaluate_realized_portfolio(
            realized, {"A": 0.5, "B": 0.5}, turnover=1.0, transaction_cost_bps=10
        )
        self.assertAlmostEqual(daily.loc[0, "Portfolio_Return"], 0.004)
        self.assertEqual(metrics["trading_days"], 2)
        self.assertAlmostEqual(metrics["turnover"], 1.0)

    def test_deepseek_request_explicitly_disables_thinking(self):
        options = completion_request_options(
            "deepseek-v4-flash", "system", "user", 0.3
        )
        self.assertEqual(options["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("reasoning_effort", options)
        self.assertEqual(options["response_format"], {"type": "json_object"})

    def test_repeated_calls_average_oriented_probabilities(self):
        predictions = [
            {"preferred_asset": "A", "probability": 0.9, "evidence": ["first"]},
            {"preferred_asset": "A", "probability": 0.6, "evidence": ["second"]},
            {"preferred_asset": "B", "probability": 0.8, "evidence": ["third"]},
        ]
        result = aggregate_repeated_predictions(predictions, "A", "B")
        self.assertAlmostEqual(result["probability_a"], (0.9 + 0.6 + 0.2) / 3)
        self.assertEqual(result["votes"], {"A": 2, "B": 1})
        np.testing.assert_allclose(result["probability_samples_a"], [0.9, 0.6, 0.2])
        self.assertAlmostEqual(result["mean_reported_probability_a"], (0.9 + 0.6 + 0.2) / 3)

    def test_preferred_asset_probability_is_normalized_to_asset_a(self):
        view = PairwiseView.from_mapping({
            "asset_a": "A",
            "asset_b": "B",
            "preferred_asset": "B",
            "probability": 0.72,
            "probability_samples": [0.70, 0.74],
        })
        self.assertAlmostEqual(view.probability_a, 0.28)
        np.testing.assert_allclose(view.probability_samples_a, [0.30, 0.26])

    def test_isotonic_predictions_are_monotone(self):
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9]
        outcomes = [0, 1, 0, 0, 1, 1, 1]
        calibrator = ProbabilityCalibrator("isotonic", min_samples=1).fit(probabilities, outcomes)
        predictions = calibrator.predict(np.linspace(0.05, 0.95, 50))
        self.assertTrue(np.all(np.diff(predictions) >= -1e-12))

    def test_abstention_and_cycle_projection_build_valid_matrices(self):
        assets = ["A", "B", "C"]
        views = [
            {"asset_a": "A", "asset_b": "B", "preferred_asset": "A", "probability": 0.8, "evidence": ["a"]},
            {"asset_a": "B", "asset_b": "C", "preferred_asset": "B", "probability": 0.75, "evidence": ["b"]},
            {"asset_a": "C", "asset_b": "A", "preferred_asset": "C", "probability": 0.7, "evidence": ["c"]},
            {"asset_a": "A", "asset_b": "D", "preferred_asset": "A", "probability": 0.55, "evidence": ["weak"]},
        ]
        config = RelViewConfig(calibration="none", abstention_threshold=0.65, max_weight=1.0)
        matrices = build_relview_matrices(assets, views, 0.02, config=config)
        self.assertEqual(matrices.P.shape, (3, 3))
        self.assertEqual(matrices.q.shape, (3,))
        self.assertEqual(matrices.omega.shape, (3, 3))
        self.assertEqual(matrices.raw_cycle_count, 1)
        self.assertEqual(len(matrices.rejected_views), 1)
        self.assertTrue(np.all(np.diag(matrices.omega) > 0))
        self.assertAlmostEqual(float(matrices.latent_scores.mean()), 0.0, places=10)

    def test_end_to_end_portfolio_is_feasible(self):
        rng = np.random.default_rng(7)
        assets = ["A", "B", "C", "D"]
        returns = pd.DataFrame(rng.normal(0.0005, 0.01, size=(80, 4)), columns=assets)
        prior = pd.Series([0.001, 0.0005, 0.0, -0.0002], index=assets)
        views = [
            {"asset_a": "A", "asset_b": "D", "preferred_asset": "A", "probability": 0.8, "evidence": ["test"]}
        ]
        result = run_relview_bl(
            returns,
            prior,
            views,
            config=RelViewConfig(calibration="none", max_weight=0.6, risk_aversion=1.0),
        )
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=8)
        self.assertTrue((result.weights >= -1e-10).all())
        self.assertTrue((result.weights <= 0.6 + 1e-10).all())
        self.assertEqual(len(result.matrices.accepted_views), 1)

    def test_realized_outcomes_are_compounded_and_oriented(self):
        views = [{
            "asset_a": "A", "asset_b": "B", "preferred_asset": "B", "probability": 0.7, "evidence": ["x"]
        }]
        realized = pd.DataFrame({"A": [0.10, -0.05], "B": [0.01, 0.01]})
        observations = calibration_observations_from_realized_returns(views, realized)
        self.assertEqual(observations[0]["outcome_a"], 1)
        self.assertAlmostEqual(observations[0]["probability_a"], 0.3)

    def test_sparse_pair_selection_prefers_sector_and_correlation(self):
        frame = pd.DataFrame({
            "A": [0.01, 0.02, -0.01, 0.03],
            "B": [0.011, 0.019, -0.009, 0.031],
            "C": [-0.02, 0.01, 0.03, -0.01],
        })
        metadata = pd.DataFrame({"Symbol": ["A", "B", "C"], "GICS Sector": ["Tech", "Tech", "Bank"]})
        pairs = select_candidate_pairs(frame, metadata=metadata, max_pairs=1)
        self.assertEqual(pairs, [("A", "B")])


if __name__ == "__main__":
    unittest.main()
