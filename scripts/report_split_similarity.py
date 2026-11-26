"""Report how similar held-out molecules are to their train split.

Usage:
    uv run python scripts/report_split_similarity.py

For each split method (scaffold, random) and seed, every val/test molecule
gets its nearest-neighbour Morgan/Tanimoto similarity to that seed's train
molecules. Writes ``data/processed/split_similarity.csv`` (gitignored) and
prints the per-method summary. This is a data-quality check, not a model
score: it shows how much harder the scaffold split is than the random one,
and how many scored molecules the demo's 0.3 out-of-domain flag would hit.
"""

import numpy as np
import pandas as pd

from tantalus.data import PROCESSED_DIR, SPLIT_FILES, load_split_table
from tantalus.domain import split_similarity
from tantalus.featurize import smiles_to_fingerprint


def main() -> None:
    """Compute similarity summaries for both split methods and save them."""
    tables = []
    for method, (splits_file, _) in SPLIT_FILES.items():
        molecules, splits = load_split_table(splits_file=splits_file)
        fingerprints = np.vstack(
            [smiles_to_fingerprint(s) for s in molecules["canonical_smiles"]]
        )
        tables.append(split_similarity(fingerprints, splits).assign(method=method))
    report = pd.concat(tables, ignore_index=True)
    report = report[["method", *report.columns.drop("method")]]
    report.to_csv(PROCESSED_DIR / "split_similarity.csv", index=False)

    summary = report.groupby(["method", "split"])[
        ["median", "p05", "frac_below_threshold"]
    ].agg(["mean", "std"])
    print("Nearest-train Tanimoto similarity, mean (std) over seeds:")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
