"""Quality gates for every artifact in the post-release multi-dataset study."""

from __future__ import annotations

import argparse
import calendar
import json
import math
from pathlib import Path

import pandas as pd

from run_multidataset_experiment import DEFAULT_MANIFEST, DEFAULT_ROOT, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--minimum-calls", type=int, default=20)
    return parser.parse_args()


def finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    config = manifest["experiment"]
    months = pd.date_range(
        f"{config['start_month']}-01", f"{config['end_month']}-01", freq="MS"
    )
    errors: list[str] = []
    absolute_files = 0
    relative_files = 0
    absolute_samples = 0
    probability_samples = 0

    release = pd.Timestamp("2026-04-24")
    if min(months) <= release:
        errors.append("formation window is not strictly after the DeepSeek V4 release")

    for dataset in manifest["datasets"]:
        dataset_id = str(dataset["id"])
        root = args.root / dataset_id
        assets = [str(item["ticker"]) for item in dataset["assets"]]
        if len(assets) != 15 or len(set(assets)) != 15:
            errors.append(f"{dataset_id}: universe must contain 15 unique assets")
        metadata = pd.read_csv(root / "metadata.csv")
        if set(metadata["Symbol"].astype(str)) != set(assets) or metadata["Name"].isna().any():
            errors.append(f"{dataset_id}: metadata does not preserve every asset identity")
        caps = json.loads((root / "equal_caps.json").read_text(encoding="utf-8"))
        if set(caps) != set(assets) or set(map(float, caps.values())) != {1.0}:
            errors.append(f"{dataset_id}: prior is not equal-cap")

        realized_months = pd.date_range(
            f"{config['start_month']}-01", f"{config['realized_end_month']}-01", freq="MS"
        )
        for month in realized_months:
            end_day = calendar.monthrange(month.year, month.month)[1]
            path = root / "returns" / f"returns_{month:%Y-%m}-01_{month:%Y-%m}-{end_day:02d}.csv"
            frame = pd.read_csv(path)
            if len(frame) < int(config["evaluation_days"]):
                errors.append(f"{path}: too few return rows")
            values = frame[assets].to_numpy(dtype=float).ravel().tolist()
            if not finite(values):
                errors.append(f"{path}: non-finite returns")

        for month in months:
            end_day = calendar.monthrange(month.year, month.month)[1]
            absolute_path = (
                root / "responses_absolute"
                / f"deepseek-v4-flash_{month:%Y-%m}-01_{month:%Y-%m}-{end_day:02d}.json"
            )
            response = json.loads(absolute_path.read_text(encoding="utf-8"))
            absolute_files += 1
            if set(response) != set(assets):
                errors.append(f"{absolute_path}: wrong asset keys")
            for asset in assets:
                samples = response.get(asset, {}).get("expected_return", [])
                absolute_samples += len(samples)
                if len(samples) < args.minimum_calls or not finite(samples):
                    errors.append(f"{absolute_path}: invalid samples for {asset}")

            relative_path = (
                root / "responses_relative" / f"deepseek-v4-flash_{month:%Y-%m}.json"
            )
            payload = json.loads(relative_path.read_text(encoding="utf-8"))
            views = payload.get("views", [])
            relative_files += 1
            if payload.get("thinking") != "disabled" or payload.get("model") != "deepseek-v4-flash":
                errors.append(f"{relative_path}: wrong model or thinking setting")
            if len(views) != int(config["max_pairs"]):
                errors.append(f"{relative_path}: wrong pair count")
            for view in views:
                samples = view.get("probability_samples_a", [])
                probability_samples += len(samples)
                valid_probability = finite(samples) and all(0.0 <= float(x) <= 1.0 for x in samples)
                if (
                    view.get("status") != "ok"
                    or int(view.get("successful_repeats", 0)) < args.minimum_calls
                    or len(samples) < args.minimum_calls
                    or not valid_probability
                ):
                    errors.append(
                        f"{relative_path}: invalid pair {view.get('asset_a')}/{view.get('asset_b')}"
                    )

        for result_name, calibration in (
            ("comparison", "isotonic"),
            ("comparison_no_calibration", "none"),
        ):
            summary_path = root / "results" / f"{result_name}_summary.json"
            periods_path = root / "results" / f"{result_name}_periods.csv"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            periods = pd.read_csv(periods_path)
            expected_methods = {
                "MVO", "BL_No_Views", "Absolute_LLM_BLM", "RelView_BL", "Equal_Weight"
            }
            if (
                summary["config"]["calibration"] != calibration
                or len(periods) != len(months)
                or set(summary["summary"]) != expected_methods
            ):
                errors.append(f"{result_name} results invalid for {dataset_id}")

    report = {
        "dataset_count": len(manifest["datasets"]),
        "assets_per_dataset": 15,
        "absolute_response_files": absolute_files,
        "relative_response_files": relative_files,
        "absolute_samples": absolute_samples,
        "probability_samples": probability_samples,
        "validation_errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
