"""NAV comparison plots for the 2025 walk-forward across 4 NVIDIA NIM models.

For each dataset and holding period (30 / 60 days), overlay the full-2025 NAV of
the two LLM methods (PairBL solid, LLM-BLM dashed) for GPT-OSS-20B, GPT-OSS-120B,
Llama-3.1-8B and Llama-3.1-70B, plus the deterministic BL baseline as a gray
reference. Colors follow a validated categorical palette in fixed order.

Run from the repository root:
    py plot_nav_4models.py
    py plot_nav_4models.py --output-dir results/nav_4models_2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODELS = [
    ("GPT-OSS-20B", "experiments/nvidia_nim_2025_walkforward", "gpt_oss_20b"),
    ("GPT-OSS-120B", "experiments/nvidia_nim_2025_walkforward_120b", "gpt_oss_120b"),
    ("Llama-3.1-8B", "experiments/nvidia_nim_2025_walkforward_llama_31_8b", "llama_31_8b"),
    ("Llama-3.1-70B", "experiments/nvidia_nim_2025_walkforward_llama_31_70b", "llama_31_70b"),
]
DATASETS = ["us_technology", "us_financials", "cross_asset_etfs"]
DATASET_LABELS = {
    "us_technology": "US Technology Equities",
    "us_financials": "US Financial Equities",
    "cross_asset_etfs": "Cross-Asset ETFs",
}
HOLDING_PERIODS = [30, 60]
# Validated categorical palette (dataviz reference), fixed order; yellow slot
# skipped because slot-4 yellow sits too close to slot-2 orange on light.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]
BL_COLOR = "#898781"
INK = "#0b0b0b"
MUTED = "#898781"


def load_daily(experiment_root: Path, dataset: str, holding_days: int) -> pd.DataFrame | None:
    path = experiment_root / dataset / f"holding_{holding_days}" / "daily_nav.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["Date"])


def nav_columns(slug: str) -> tuple[str, str]:
    return f"RelViewBL__{slug}_NAV", f"BLM_LLM__{slug}_NAV"


def plot_dataset_holding(
    dataset: str, holding_days: int, output_dir: Path,
) -> list[tuple[str, pd.DataFrame]]:
    fig, ax = plt.subplots(figsize=(14, 7))
    loaded: list[tuple[str, pd.DataFrame]] = []
    for index, (label, root, slug) in enumerate(MODELS):
        daily = load_daily(Path(root), dataset, holding_days)
        if daily is None or daily.empty:
            continue
        loaded.append((label, daily))
        color = PALETTE[index % len(PALETTE)]
        pair_col, llm_col = nav_columns(slug)
        if pair_col in daily.columns:
            ax.plot(daily["Date"], daily[pair_col], color=color, linewidth=2.2,
                    label=f"PairBL ({label})")
        if llm_col in daily.columns:
            ax.plot(daily["Date"], daily[llm_col], color=color, linewidth=1.4,
                    linestyle=(0, (4, 2)), label=f"LLM-BLM ({label})")
    # BL baseline is model-independent; plot from the first loaded frame if any.
    if loaded:
        _, first = loaded[0]
        if "BL_NAV" in first.columns:
            ax.plot(first["Date"], first["BL_NAV"], color=BL_COLOR, linewidth=1.2,
                    linestyle=(0, (1, 1)), label="BL baseline")

    ax.axhline(1.0, color=BL_COLOR, linewidth=0.7, alpha=0.5)
    ax.set_title(
        f"{DATASET_LABELS.get(dataset, dataset)} — NAV 2025 | "
        f"{holding_days}-Day Rebalance | 4-Model Comparison",
        fontsize=14, fontweight="bold", color=INK,
    )
    ax.set_xlabel("Date", color=MUTED)
    ax.set_ylabel("NAV (initial capital = 1)", color=MUTED)
    ax.grid(alpha=0.25, color="#e1e0d9")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{dataset}_holding_{holding_days}_nav.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    return loaded


def write_combined_csv(loaded: list[tuple[str, pd.DataFrame]], dataset: str,
                       holding_days: int, output_dir: Path) -> None:
    if not loaded:
        return
    slug_by_label = {label: slug for label, _, slug in MODELS}
    combined = None
    for index, (label, daily) in enumerate(loaded):
        pair_col, llm_col = nav_columns(slug_by_label[label])
        columns = [pair_col, llm_col]
        new_names = [f"PairBL_{label}", f"LLM_BLM_{label}"]
        # BL is model-independent; carry it from the first loaded frame only.
        if index == 0 and "BL_NAV" in daily.columns:
            columns.append("BL_NAV")
            new_names.append("BL")
        sub = daily[["Date"] + columns].copy()
        sub.columns = ["Date"] + new_names
        combined = sub if combined is None else combined.merge(sub, on="Date", how="outer")
    if combined is not None:
        combined = combined.sort_values("Date")
        out = output_dir / f"{dataset}_holding_{holding_days}_nav.csv"
        combined.to_csv(out, index=False)
        print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/nav_4models_2025"))
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--holding-periods", nargs="+", type=int, default=HOLDING_PERIODS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        for holding_days in args.holding_periods:
            loaded = plot_dataset_holding(dataset, holding_days, args.output_dir)
            write_combined_csv(loaded, dataset, holding_days, args.output_dir)
    print(f"All NAV plots under {args.output_dir}")


if __name__ == "__main__":
    main()
