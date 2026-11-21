"""Dataset loading, cleaning, merging and splits.

Every loader returns two tables: the rows it kept and the rows it dropped
(with a ``drop_reason`` column), so every drop can be counted and documented.
"""

from pathlib import Path

import pandas as pd

from tantalus.featurize import canonicalize

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CHEMTASTESDB_XLSX = RAW_DIR / "chemtastesdb" / "ChemTastesDB_database.xlsx"

# Original ChemTastesDB headers -> snake_case names used in this project.
_CHEMTASTESDB_COLUMNS = {
    "ID": "source_id",
    "Name": "name",
    "PubChem CID": "pubchem_cid",
    "CAS number": "cas",
    "canonical SMILES": "source_smiles",
    "Taste": "taste",
    "Class taste": "class_taste",
    "Reference_(cod)/[pp]": "references",
}


def _split_drops(
    df: pd.DataFrame, mask: pd.Series, reason: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (rows where mask is False, rows where mask is True + reason)."""
    dropped = df[mask].assign(drop_reason=reason)
    return df[~mask], dropped


def load_chemtastesdb(
    path: Path = CHEMTASTESDB_XLSX,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ChemTastesDB and add an RDKit ``canonical_smiles`` column.

    The file's own "canonical SMILES" column was made by a different toolkit
    (Marvin), so it is kept as ``source_smiles`` and re-canonicalized with
    RDKit. That way, every source uses the same key for dedup and splitting.

    No dedup, salt stripping, or label mapping happens here. Those are
    separate Phase 1 steps.

    Args:
        path: Path to ``ChemTastesDB_database.xlsx``.

    Returns:
        ``(kept, dropped)``. ``kept`` has the renamed columns plus
        ``canonical_smiles`` and ``source="chemtastesdb"``. ``dropped`` has
        the same columns plus a ``drop_reason`` for each removed row.
    """
    # dtype=str keeps zero-padded IDs ("0001") and CAS numbers intact.
    df = pd.read_excel(path, dtype=str).rename(columns=_CHEMTASTESDB_COLUMNS)
    df["source"] = "chemtastesdb"

    # The sheet ends with footnote rows that have no molecule at all.
    df, no_smiles = _split_drops(df, df["source_smiles"].isna(), "no SMILES")

    df = df.assign(canonical_smiles=df["source_smiles"].map(canonicalize))
    df, invalid = _split_drops(
        df, df["canonical_smiles"].isna(), "SMILES not parsable by RDKit"
    )

    dropped = pd.concat([no_smiles, invalid], ignore_index=True)
    return df.reset_index(drop=True), dropped
