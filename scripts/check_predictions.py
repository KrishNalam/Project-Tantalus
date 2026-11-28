"""Check every saved prediction file against the data it claims to score.

Usage:
    uv run python scripts/check_predictions.py

This is an integrity check, not a metric: it computes no scores (those come
from the owner's ``evaluate.py``). For each ``data/processed/predictions/*.csv``
it verifies, per seed, that:

- the columns are the shared format (SMILES, seed, split, ``<label>_true`` /
  ``<label>_prob`` for every label);
- the scored rows are exactly that seed's val + test molecules in the split
  file the predictions came from (``*_random.csv`` -> ``random_splits.csv``,
  otherwise ``splits.csv``): no train molecule (leakage), none missing, no
  duplicates, and each row's ``split`` matches the split file;
- every ``<label>_true`` equals the label in ``molecules.csv``;
- every probability is finite and in [0, 1].

It also counts probabilities saturated to exactly 0.0 or 1.0, which create
ties in ROC/PR-AUC. Exits with status 1 if any file fails.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tantalus.data import LABELS, PROCESSED_DIR, SPLIT_FILES, load_split_table

PREDICTIONS_DIR = PROCESSED_DIR / "predictions"
EXPECTED_COLUMNS = ["canonical_smiles", "seed", "split"] + [
    f"{label}_{kind}" for label in LABELS for kind in ("true", "prob")
]


def check_file(
    path: Path, molecules: pd.DataFrame, splits: pd.DataFrame
) -> tuple[list[str], int]:
    """Check one prediction file against its molecules table and split file.

    Args:
        path: A ``predictions/<model>.csv`` file.
        molecules: ``molecules.csv`` as returned by ``load_split_table``.
        splits: The matching split assignments, row-aligned with ``molecules``.

    Returns:
        ``(problems, n_saturated)``: human-readable problem descriptions (empty
        if the file is clean) and the number of probabilities equal to exactly
        0.0 or 1.0.
    """
    preds = pd.read_csv(path)
    if list(preds.columns) != EXPECTED_COLUMNS:
        return [f"columns differ from the shared format: {list(preds.columns)}"], 0

    problems = []
    if preds.duplicated(["canonical_smiles", "seed"]).any():
        problems.append("duplicate (molecule, seed) rows")

    for seed, seed_splits in splits.drop(columns="canonical_smiles").items():
        k = int(seed.removeprefix("seed_"))
        expected = pd.DataFrame(
            {"canonical_smiles": splits["canonical_smiles"], "split": seed_splits}
        )
        expected = expected[expected["split"] != "train"]
        got = preds.loc[preds["seed"] == k, ["canonical_smiles", "split"]]
        merged = expected.merge(got, on="canonical_smiles", how="outer", indicator=True)
        missing = (merged["_merge"] == "left_only").sum()
        extra = (merged["_merge"] == "right_only").sum()
        wrong = (
            (merged["_merge"] == "both") & (merged["split_x"] != merged["split_y"])
        ).sum()
        if missing or extra or wrong:
            problems.append(
                f"seed {k}: {missing} val/test molecules missing, {extra} not in "
                f"val/test (train leakage or unknown SMILES), {wrong} with the wrong split"
            )

    truth = preds[["canonical_smiles"]].merge(
        molecules[["canonical_smiles", *LABELS]], on="canonical_smiles", how="left"
    )
    for label in LABELS:
        if not np.array_equal(
            preds[f"{label}_true"].to_numpy(), truth[label].to_numpy()
        ):
            problems.append(f"{label}_true disagrees with molecules.csv")

    probs = preds[[f"{label}_prob" for label in LABELS]].to_numpy()
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        problems.append("probabilities not all finite and in [0, 1]")
    n_saturated = int(((probs == 0.0) | (probs == 1.0)).sum())
    return problems, n_saturated


def main() -> None:
    """Check every prediction file and print a one-line verdict per file."""
    tables = {
        method: load_split_table(splits_file=splits_file)
        for method, (splits_file, _) in SPLIT_FILES.items()
    }
    failed = False
    for path in sorted(PREDICTIONS_DIR.glob("*.csv")):
        method = "random" if path.stem.endswith("_random") else "scaffold"
        problems, n_saturated = check_file(path, *tables[method])
        verdict = "FAIL" if problems else "ok"
        print(f"{verdict:4}  {path.name:24} {method:8} saturated probs: {n_saturated}")
        for problem in problems:
            print(f"      - {problem}")
        failed |= bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
