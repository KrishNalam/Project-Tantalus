"""Build the processed dataset and its scaffold splits in data/processed/.

Usage:
    uv run python scripts/download_data.py   # once
    uv run python scripts/build_data.py

Writes molecules.csv, dropped.csv, salt_log.csv, merge_log.csv, splits.csv
and split_balance.csv (all gitignored), then prints the class balance:
the minimum number of positives per (split, label) over all split seeds.
A label with 0 there would get an undefined ROC-AUC for some seed.
"""

from rdkit import RDLogger

from tantalus.data import build_dataset, write_splits


def main() -> None:
    """Run the Phase 1 pipeline and print a class-balance summary."""
    RDLogger.DisableLog("rdApp.*")  # parse warnings are already logged as drops
    molecules = build_dataset()
    _, balance = write_splits(molecules)
    print(f"{len(molecules)} molecules")
    summary = balance.pivot_table(
        index="label", columns="split", values="n_positive", aggfunc="min"
    )
    print("min positives per split over seeds:")
    print(summary[["train", "val", "test"]].to_string())


if __name__ == "__main__":
    main()
