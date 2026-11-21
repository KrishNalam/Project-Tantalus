"""Dataset loading, cleaning, merging and splits.

Every loader returns two tables: the rows it kept and the rows it dropped
(with a ``drop_reason`` column), so every drop can be counted and documented.
"""

from pathlib import Path

import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

from tantalus.featurize import _parse, canonicalize

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CHEMTASTESDB_XLSX = RAW_DIR / "chemtastesdb" / "ChemTastesDB_database.xlsx"
BITTERDB_CSV = RAW_DIR / "bitterdb" / "BitterCompoundsPropA_2019.csv"

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

# BitterDB columns we keep (most of its 30 are descriptors we recompute or don't
# need) -> snake_case names matching the ChemTastesDB loader where possible.
_BITTERDB_COLUMNS = {
    "cid": "source_id",
    "pubChemID": "pubchem_cid",
    "Cas_Number_Final": "cas",
    "InChiKey": "inchikey",
    "canonical_smiles": "source_smiles",
    "Human": "bitter_human",
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


def load_bitterdb(path: Path = BITTERDB_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the BitterDB 2019 compound table and add RDKit ``canonical_smiles``.

    Uses BitterDB's stereo-free ``canonical_smiles`` column, not
    ``Isomeric_smiles``, because ChemTastesDB has no stereochemistry either.
    Mixing the two would make the same molecule look different across sources
    and break dedup.

    BitterDB records bitterness per species. Rows that are not bitter to
    humans (``Human == "0"``, e.g. mouse-only) are dropped, because this
    project predicts human taste. No dedup or salt stripping happens here.

    Args:
        path: Path to ``BitterCompoundsPropA_2019.csv``.

    Returns:
        ``(kept, dropped)`` with the kept columns renamed, plus
        ``canonical_smiles`` and ``source="bitterdb"``. ``dropped`` has the
        same columns plus a ``drop_reason``.
    """
    df = pd.read_csv(path, dtype=str, usecols=list(_BITTERDB_COLUMNS))
    df = df.rename(columns=_BITTERDB_COLUMNS)
    df["source"] = "bitterdb"

    df, not_human = _split_drops(
        df, df["bitter_human"] != "1", "not bitter to humans (other species only)"
    )
    df, no_smiles = _split_drops(df, df["source_smiles"].isna(), "no SMILES")

    df = df.assign(canonical_smiles=df["source_smiles"].map(canonicalize))
    df, invalid = _split_drops(
        df, df["canonical_smiles"].isna(), "SMILES not parsable by RDKit"
    )

    dropped = pd.concat([not_human, no_smiles, invalid], ignore_index=True)
    return df.drop(columns="bitter_human").reset_index(drop=True), dropped


def murcko_scaffold(smiles: str) -> str | None:
    """Return the Bemis-Murcko scaffold of a molecule as canonical SMILES.

    The scaffold is the molecule's ring systems plus the linker atoms that
    connect them, with every side chain removed. Molecules that share a
    scaffold are close analogues, so the scaffold split keeps each scaffold
    group entirely in train, validation or test.

    This is the specific (atom- and bond-typed) scaffold, not the generic
    "all carbon, all single bonds" version. Stereochemistry is ignored, to
    match the stereo-free SMILES used everywhere else.

    Args:
        smiles: A SMILES string, ideally already canonical.

    Returns:
        The scaffold's canonical SMILES. Acyclic molecules have no rings and
        return ``""``, so they all form one group. Returns ``None`` if the
        SMILES cannot be parsed. For multi-fragment inputs (salts), ring-free
        fragments such as ``[Na+]`` disappear and ring-containing fragments
        are kept, joined by ``.``.
    """
    mol = _parse(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
