"""Train the Random Forest baseline on every saved scaffold split.

Usage:
    uv run python scripts/build_data.py         # once, writes the splits
    uv run python scripts/run_random_forest.py

For each split seed, fits on ``train`` only and writes val + test
probabilities to ``data/processed/predictions/random_forest.csv`` (gitignored).
Columns: canonical_smiles, seed, split, then ``<label>_true`` and
``<label>_prob`` for every label. Scoring is done by ``evaluate.py``.
"""

import pandas as pd

from tantalus.data import LABELS, PROCESSED_DIR
from tantalus.models.baselines import (
    load_fingerprint_table,
    predict_proba,
    train_random_forest,
)


def main() -> None:
    """Fit one forest per (seed, label) and save val/test predictions."""
    X, Y, splits = load_fingerprint_table()
    seed_columns = [c for c in splits.columns if c.startswith("seed_")]
    frames = []
    for column in seed_columns:
        seed = int(column.removeprefix("seed_"))
        split = splits[column].to_numpy()
        train = split == "train"
        models = train_random_forest(X[train], Y[train], seed=seed)
        evaluated = split != "train"
        probs = predict_proba(models, X[evaluated])
        frame = pd.DataFrame(
            {
                "canonical_smiles": splits["canonical_smiles"][evaluated].to_numpy(),
                "seed": seed,
                "split": split[evaluated],
            }
        )
        for j, label in enumerate(LABELS):
            frame[f"{label}_true"] = Y[evaluated, j]
            frame[f"{label}_prob"] = probs[:, j]
        frames.append(frame)
        print(f"seed {seed}: trained on {train.sum()}, predicted {evaluated.sum()}")

    out_dir = PROCESSED_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "random_forest.csv"
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
