"""Dataset loading, cleaning, merging and splits.

Every loader returns two tables: the rows it kept and the rows it dropped
(with a ``drop_reason`` column), so every drop can be counted and documented.
"""

import math
import random
from collections import defaultdict
from collections.abc import Sequence
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


def scaffold_split(
    smiles: Sequence[str],
    frac: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """Split molecules into train/val/test so that no scaffold spans two splits.

    This is the "balanced" scaffold split used by Chemprop. Molecules are
    grouped by :func:`murcko_scaffold`. Groups bigger than half the val or test
    target size are "big" and are placed first, so they always land in train
    (otherwise one huge group could fill the whole test set). The remaining
    small groups are shuffled with ``seed``, which gives a different but still
    scaffold-disjoint split per seed. Groups are then assigned greedily: train
    while it has room, then val, then test.

    Duplicate SMILES share a scaffold, so they always land in the same split.

    Args:
        smiles: SMILES strings, one per row (ideally canonical and cleaned).
        frac: Target (train, val, test) fractions. Must sum to 1. Real sizes
            differ slightly because whole groups are assigned at once.
        seed: Seed for shuffling the group order.

    Returns:
        ``(train_idx, val_idx, test_idx)``: positional indices into ``smiles``.

    Raises:
        ValueError: If ``frac`` doesn't sum to 1, or a SMILES can't be parsed.
            Unparsable rows should be dropped (and logged) by the loaders, so
            reaching this point means a bug upstream, not a row to skip.
    """
    if not math.isclose(sum(frac), 1.0):
        raise ValueError(f"frac must sum to 1, got {frac}")

    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        scaffold = murcko_scaffold(smi)
        if scaffold is None:
            raise ValueError(f"cannot compute scaffold for row {i}: {smi!r}")
        groups[scaffold].append(i)

    n = len(smiles)
    train_size, val_size, test_size = (f * n for f in frac)
    big_cutoff = min(val_size, test_size) / 2
    big = [g for g in groups.values() if len(g) > big_cutoff]
    small = [g for g in groups.values() if len(g) <= big_cutoff]

    rng = random.Random(seed)
    rng.shuffle(big)
    rng.shuffle(small)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for group in big + small:
        if len(train) + len(group) <= train_size:
            train.extend(group)
        elif len(val) + len(group) <= val_size:
            val.extend(group)
        else:
            test.extend(group)
    return sorted(train), sorted(val), sorted(test)


def class_balance(
    labels: pd.DataFrame, splits: dict[str, Sequence[int]]
) -> pd.DataFrame:
    """Count positives per label in each split (the Phase 1 class-balance report).

    Works on any multi-hot label table, so it doesn't depend on how the raw
    taste strings are mapped to labels. A label with 0 positives in a split
    is worth catching early: ROC-AUC and PR-AUC are undefined for it there.

    Args:
        labels: One row per molecule, one 0/1 column per label (e.g. "sweet").
            Rows are addressed by position, matching :func:`scaffold_split`.
        splits: Split name -> positional row indices, e.g.
            ``{"train": train_idx, "val": val_idx, "test": test_idx}``.

    Returns:
        A long table with one row per (split, label) and the columns
        ``split``, ``label``, ``n_molecules``, ``n_positive``, ``frac_positive``.

    Raises:
        ValueError: If a label column isn't 0/1, or a row appears in more than
            one split (that would leak the same molecule into train and test).
    """
    values = labels.to_numpy()
    if not ((values == 0) | (values == 1)).all():
        raise ValueError("labels must be multi-hot (only 0/1 values)")

    seen: set[int] = set()
    for name, idx in splits.items():
        overlap = seen.intersection(idx)
        if overlap:
            raise ValueError(f"{len(overlap)} rows of {name!r} are in another split")
        seen.update(idx)

    rows = []
    for name, idx in splits.items():
        part = labels.iloc[list(idx)]
        for label in labels.columns:
            n_pos = int(part[label].sum())
            rows.append(
                {
                    "split": name,
                    "label": label,
                    "n_molecules": len(part),
                    "n_positive": n_pos,
                    "frac_positive": n_pos / len(part) if len(part) else float("nan"),
                }
            )
    return pd.DataFrame(rows)
