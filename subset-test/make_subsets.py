"""Build random and coverage-aware ("smart") training subsets of GEP-M.

The question this answers: if we only keep a fraction of GEP-M, does picking
series by their statistical profile beat picking them at random?

Both methods take the same number of series from every source, so the source
mix is identical. The only difference is which series get picked inside each
source:

    random  picks uniformly at random.
    smart   first takes one series from every statistical profile present in
            the source, then fills the rest of the source's quota at random.

Profiles use the same recipe as the statistical sampler on the
statistical-data-sampling branch: four features (relative IQR, skewness,
excess kurtosis, lag-1 autocorrelation), each cut into five global quantile
bins. Series with a missing feature share one extra "missing" profile.

Each output folder holds train-*.parquet (the subset) and validation-*.parquet
(the full GEP-M validation split, unchanged), so train.py can use it directly:

    uv run python -m nanotsfm.train --config configs/series.yaml \
        --output runs/random-25-s7 --data subsets/random-25-s7 --seed 7

Example:
    uv run python subset-test/make_subsets.py \
        --stats statistical-approach/series_stats_GEP-M_train.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nanotsfm.data import load

FEATURES = ["relative_iqr", "skewness", "kurtosis", "autocorrelation"]
MISSING_PROFILE = (-1, -1, -1, -1)


def assign_profiles(stats: pd.DataFrame, bins: int) -> pd.Series:
    """Give every series a profile: one quantile bin per feature."""
    values = stats[FEATURES].replace([np.inf, -np.inf], np.nan)
    complete = values.notna().all(axis=1)

    profile_bins = pd.DataFrame(index=stats.index)
    for feature in FEATURES:
        # Rank only the complete rows, matching the sampler on the branch.
        ranks = values.loc[complete, feature].rank(pct=True)
        profile_bins[feature] = np.minimum((ranks * bins).astype(int), bins - 1)

    profiles = []
    for row_label, is_complete in complete.items():
        if is_complete:
            profiles.append(tuple(int(profile_bins.at[row_label, feature]) for feature in FEATURES))
        else:
            profiles.append(MISSING_PROFILE)
    return pd.Series(profiles, index=stats.index, dtype=object)


def pick_random(rows: pd.DataFrame, quota: int, rng) -> list[int]:
    return rng.choice(rows["dataset_index"].to_numpy(), size=quota, replace=False).tolist()


def pick_smart(rows: pd.DataFrame, quota: int, rng) -> list[int]:
    """One series per profile first, then fill the quota at random."""
    groups = [group["dataset_index"].to_numpy() for _, group in rows.groupby("profile")]

    # A small source can have more profiles than its quota allows.
    # Then cover as many profiles as fit, chosen at random.
    if len(groups) > quota:
        chosen_groups = rng.choice(len(groups), size=quota, replace=False)
        groups = [groups[i] for i in chosen_groups]

    picked = [int(rng.choice(group)) for group in groups]

    leftover = np.setdiff1d(rows["dataset_index"].to_numpy(), picked)
    fill = quota - len(picked)
    if fill > 0:
        picked += rng.choice(leftover, size=fill, replace=False).tolist()
    return picked


def build_subset(stats: pd.DataFrame, method: str, fraction: float, seed: int) -> list[int]:
    picked = []
    # Sorting the sources keeps the result the same on every machine.
    for source_number, source in enumerate(sorted(stats["source"].unique())):
        rows = stats[stats["source"] == source]
        quota = max(1, round(fraction * len(rows)))
        rng = np.random.default_rng([seed, source_number])
        if method == "random":
            picked += pick_random(rows, quota, rng)
        else:
            picked += pick_smart(rows, quota, rng)
    return sorted(int(i) for i in picked)


def describe_subset(stats: pd.DataFrame, picked: list[int]) -> dict:
    """How many (source, profile) buckets the subset still covers."""
    all_buckets = set(zip(stats["source"], stats["profile"]))
    chosen = stats[stats["dataset_index"].isin(picked)]
    kept_buckets = set(zip(chosen["source"], chosen["profile"]))
    return {
        "series": len(picked),
        "buckets_total": len(all_buckets),
        "buckets_covered": len(kept_buckets),
        "bucket_coverage": round(len(kept_buckets) / len(all_buckets), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", type=Path, required=True, help="series_stats_GEP-M_train.parquet")
    parser.add_argument("--data", default="GEP-M", help="dataset the stats were computed on")
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--bins", type=int, default=5, help="quantile bins per feature")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=["random", "smart"], default=["random", "smart"])
    parser.add_argument("--output", type=Path, default=Path("subsets"))
    parser.add_argument("--dry-run", action="store_true", help="print the summary without writing parquet")
    args = parser.parse_args()

    if not 0 < args.fraction < 1:
        parser.error("--fraction must be between 0 and 1")

    stats = pd.read_parquet(args.stats)
    if stats["item_id"].duplicated().any():
        raise ValueError("The stats file has duplicate item_id values")

    # Series too short to train on are dropped by the trainer anyway.
    if "eligible_for_training" in stats.columns:
        stats = stats[stats["eligible_for_training"].fillna(False)].copy()
    stats["profile"] = assign_profiles(stats, args.bins)

    train = None
    if not args.dry_run:
        train = load(args.data, "train")
        # The stats rows point at dataset rows by index. Make sure they line up.
        check = train.select(stats["dataset_index"].tolist())["item_id"]
        if list(check) != stats["item_id"].tolist():
            raise ValueError(f"The stats file does not match {args.data} train. Was it built from another dataset?")
        validation = load(args.data, "validation")

    percent = round(args.fraction * 100)
    summary = {"data": args.data, "fraction": args.fraction, "bins": args.bins, "subsets": {}}

    for seed in args.seeds:
        picks = {}
        for method in args.methods:
            name = f"{method}-{percent}-s{seed}"
            picked = build_subset(stats, method, args.fraction, seed)
            picks[method] = set(picked)
            summary["subsets"][name] = describe_subset(stats, picked)
            print(name, summary["subsets"][name])

            if train is not None:
                folder = args.output / name
                folder.mkdir(parents=True, exist_ok=True)
                train.select(picked).to_parquet(folder / "train-00000-of-00001.parquet")
                validation.to_parquet(folder / "validation-00000-of-00001.parquet")

        if len(picks) == 2:
            shared = len(picks["random"] & picks["smart"])
            summary["subsets"][f"overlap-s{seed}"] = shared
            print(f"seed {seed}: random and smart share {shared} series")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
