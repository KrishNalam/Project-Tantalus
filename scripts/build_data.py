"""Build the processed dataset and its scaffold splits in data/processed/.

Usage:
    uv run python scripts/download_data.py   # once
    uv run python scripts/build_data.py

Writes molecules.csv, dropped.csv, salt_log.csv, merge_log.csv, the scaffold
split (splits.csv, split_balance.csv) and the random split used only as a
deliberate comparison (random_splits.csv, random_split_balance.csv), all
gitignored. Then prints the class balance of each: the minimum number of
positives per (split, label) over all split seeds. A label with 0 there
would get an undefined ROC-AUC for some seed.
"""

from rdkit import RDLogger

from tantalus.data import build_dataset, write_splits


def main() -> None:
    """Run the Phase 1 pipeline and print a class-balance summary."""
    RDLogger.DisableLog("rdApp.*")  # parse warnings are already logged as drops
    molecules = build_dataset()
    print(f"{len(molecules)} molecules")
    for method in ("scaffold", "random"):
        _, balance = write_splits(molecules, method=method)
        summary = balance.pivot_table(
            index="label", columns="split", values="n_positive", aggfunc="min"
        )
        print(f"{method} split, min positives per split over seeds:")
        print(summary[["train", "val", "test"]].to_string())


if __name__ == "__main__":
    main()
