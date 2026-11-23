"""Dataset loading, cleaning, merging and splits.

Every loader returns two tables: the rows it kept and the rows it dropped
(with a ``drop_reason`` column), so every drop can be counted and documented.
"""

import math
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from tantalus.featurize import _parse, canonicalize

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CHEMTASTESDB_XLSX = RAW_DIR / "chemtastesdb" / "ChemTastesDB_database.xlsx"
BITTERDB_CSV = RAW_DIR / "bitterdb" / "BitterCompoundsPropA_2019.csv"

# Basic tastes we look for in ChemTastesDB's free-text "Taste" column, and the
# words that count as each one. "neutral" means tasteless, "savory" means umami.
_TASTE_WORDS = {
    "sweet": ["sweet"],
    "bitter": ["bitter"],
    "umami": ["umami", "savory"],
    "sour": ["sour"],
    "salty": ["salty"],
    "tasteless": ["tasteless", "neutral"],
}
# The six output labels, in the column order used everywhere downstream.
LABELS = ["sweet", "bitter", "umami", "sour", "salty", "tasteless"]

# ChemTastesDB classes that name exactly one label. "Non-sweetness" and
# "Non-bitterness" map to no positive label (all zeros).
_CLASS_TO_LABEL = {
    "Sweetness": "sweet",
    "Bitterness": "bitter",
    "Umaminess": "umami",
    "Sourness": "sour",
    "Saltiness": "salty",
    "Tastelessness": "tasteless",
    "Non-sweetness": None,
    "Non-bitterness": None,
}
# Catch-all classes whose labels are read from the free-text "taste" column.
_FREE_TEXT_CLASSES = {"Multitaste", "Miscellaneous"}

# Explicit negations, kept separately because ChemTastesDB has whole classes of
# them (Non-sweetness, Non-bitterness).
_NEGATED_WORDS = {"non_sweet": "sweet", "non_bitter": "bitter"}


def _word_pattern(words: list[str]) -> re.Pattern[str]:
    """Match any of ``words`` as a whole word, not preceded by ``-`` (so ``non-sweet``
    doesn't count as ``sweet``) and not followed by a letter (so ``umamiless``
    doesn't count as ``umami``)."""
    return re.compile(r"(?<![\w-])(?:" + "|".join(words) + r")(?!\w)", re.IGNORECASE)


_TASTE_PATTERNS = {name: _word_pattern(w) for name, w in _TASTE_WORDS.items()}
_NEGATED_PATTERNS = {
    name: re.compile(r"(?<![\w-])no[nt]-" + word + r"(?!\w)", re.IGNORECASE)
    for name, word in _NEGATED_WORDS.items()
}

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


def split_coverage(
    smiles: Sequence[str],
    labels: pd.DataFrame,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    frac: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> pd.DataFrame:
    """Report positives per (seed, split, label) across several scaffold splits.

    :func:`class_balance` checks one split. Reported results are mean +/- std
    over seeds, and each seed gives a different scaffold split, so a label is
    only evaluable if val *and* test have positives for *every* seed. This
    runs :func:`scaffold_split` once per seed and stacks the class-balance
    tables, so ``min(n_positive)`` per (split, label) shows which labels
    would get an undefined ROC-AUC for some seed.

    Args:
        smiles: SMILES strings, one per row of ``labels`` (same order).
        labels: Multi-hot table, one 0/1 column per label.
        seeds: Split seeds to try.
        frac: Target (train, val, test) fractions, passed to
            :func:`scaffold_split`.

    Returns:
        The :func:`class_balance` columns plus a leading ``seed`` column,
        one row per (seed, split, label).

    Raises:
        ValueError: If ``smiles`` and ``labels`` have different lengths, or
            for any reason :func:`scaffold_split` or :func:`class_balance`
            raises.
    """
    if len(smiles) != len(labels):
        raise ValueError(f"{len(smiles)} SMILES but {len(labels)} label rows")
    tables = []
    for seed in seeds:
        train, val, test = scaffold_split(smiles, frac=frac, seed=seed)
        table = class_balance(labels, {"train": train, "val": val, "test": test})
        tables.append(table.assign(seed=seed))
    report = pd.concat(tables, ignore_index=True)
    return report[["seed", *report.columns.drop("seed")]]


def label_conflicts(
    df: pd.DataFrame,
    label_col: str,
    key: str = "canonical_smiles",
) -> pd.DataFrame:
    """List molecules whose duplicate rows disagree on a label (a dedup report).

    This only *finds* conflicts; it doesn't resolve them. How to resolve them
    (drop, majority vote, prefer a source, union into multi-hot) is a separate
    decision, and this report is the evidence for it. It works within one
    source or across several stacked with ``pd.concat``, because it groups by
    the RDKit canonical SMILES that every loader adds.

    Args:
        df: Rows from one or more loaders. Needs ``key``, ``label_col`` and
            ``source`` columns.
        label_col: Column with the raw label to compare, e.g. ``"class_taste"``.
            Missing values are ignored.
        key: Column that identifies a molecule. Defaults to the canonical SMILES.

    Returns:
        One row per conflicting molecule, sorted by ``key``, with the columns
        ``<key>``, ``n_rows`` (rows with a non-missing label), ``n_labels``,
        ``labels`` and ``sources`` (the last two are sorted and joined with
        ``" | "``). Empty if nothing conflicts.
    """
    rows = df.dropna(subset=[label_col])
    grouped = rows.groupby(key).agg(
        n_rows=(label_col, "size"),
        n_labels=(label_col, "nunique"),
        labels=(label_col, lambda s: " | ".join(sorted(s.unique()))),
        sources=("source", lambda s: " | ".join(sorted(s.unique()))),
    )
    conflicts = grouped[grouped["n_labels"] > 1]
    return conflicts.sort_index().reset_index()


def salt_report(
    df: pd.DataFrame,
    label_col: str,
    key: str = "canonical_smiles",
) -> pd.DataFrame:
    """List multi-fragment molecules (salts, mixtures) as evidence for a salt rule.

    This only *reports*; it doesn't strip anything. For each row whose SMILES
    has more than one ``.``-separated fragment, it names the "parent" (the
    fragment with the most heavy atoms, which largest-fragment stripping
    would keep) and the fragments that stripping would throw away. The
    ``parent_listed`` flag shows whether the parent also appears as its own
    row: those pairs are near-duplicates that could land on opposite sides of
    a split if left as two molecules (a leakage risk).

    Args:
        df: Rows from one or more loaders. Needs ``key``, ``label_col`` and
            ``source`` columns.
        label_col: Column with the raw label, e.g. ``"class_taste"``, so the
            report can show which labels (salty, umami) depend on the ion.
        key: Column holding the canonical SMILES. Defaults to
            ``"canonical_smiles"``.

    Returns:
        One row per multi-fragment input row, sorted by ``key``, with the
        columns ``<key>``, ``n_fragments``, ``parent_smiles``,
        ``other_fragments`` (sorted, joined with ``" | "``), ``parent_listed``,
        ``<label_col>`` and ``source``. Empty if nothing is multi-fragment.
        Rows whose SMILES can't be parsed are skipped.
    """
    known = set(df[key].dropna())
    records = []
    for smiles, label, source in zip(df[key], df[label_col], df["source"]):
        mol = _parse(smiles)
        if mol is None:
            continue
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) < 2:
            continue
        frag_smiles = [Chem.MolToSmiles(f) for f in frags]
        # Most heavy atoms wins; ties broken by SMILES so the choice is stable.
        parent_idx = max(
            range(len(frags)),
            key=lambda i: (frags[i].GetNumHeavyAtoms(), frag_smiles[i]),
        )
        parent = frag_smiles[parent_idx]
        others = sorted(s for i, s in enumerate(frag_smiles) if i != parent_idx)
        records.append(
            {
                key: smiles,
                "n_fragments": len(frags),
                "parent_smiles": parent,
                "other_fragments": " | ".join(others),
                "parent_listed": parent in known,
                label_col: label,
                "source": source,
            }
        )
    columns = [
        key,
        "n_fragments",
        "parent_smiles",
        "other_fragments",
        "parent_listed",
        label_col,
        "source",
    ]
    report = pd.DataFrame(records, columns=columns)
    return report.sort_values(key, kind="stable").reset_index(drop=True)


def taste_mentions(df: pd.DataFrame, taste_col: str = "taste") -> pd.DataFrame:
    """Flag which basic tastes each row's free-text description mentions.

    ChemTastesDB's ``class_taste`` lumps rows into catch-all classes
    (Multitaste, Miscellaneous), but its free-text ``taste`` column says which
    tastes were reported, e.g. ``"Bitter; Umami"``. This report reads that
    text so the owner can decide the label mapping from evidence. It doesn't
    *apply* a mapping: the flags are keyword matches, and words like
    ``"umami-like"`` or ``"umami-enhancing"`` still count as ``umami``.

    Args:
        df: Rows with a free-text taste column (e.g. from
            :func:`load_chemtastesdb`). Missing values give all-False rows.
        taste_col: Name of that column. Defaults to ``"taste"``.

    Returns:
        A table with the same index as ``df`` and one boolean column per basic
        taste (``sweet``, ``bitter``, ``umami``, ``sour``, ``salty``,
        ``tasteless``) plus the explicit negations ``non_sweet`` and
        ``non_bitter``. ``"non-sweet"`` sets ``non_sweet``, not ``sweet``.
    """
    text = df[taste_col].fillna("").astype(str)
    patterns = _TASTE_PATTERNS | _NEGATED_PATTERNS
    return pd.DataFrame(
        {name: text.str.contains(pat) for name, pat in patterns.items()},
        index=df.index,
    )


def assign_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add one 0/1 column per label in :data:`LABELS` to each row (no dedup yet).

    The rules (owner decisions, see ``docs/decisions.md``):

    - BitterDB rows: ``bitter = 1``, everything else 0.
    - ChemTastesDB single-taste classes (Sweetness, Bitterness, ...): that one
      label. The curated class wins over the free text here.
    - Non-sweetness / Non-bitterness: all zeros. They are known negatives for
      one taste, not "tasteless".
    - Multitaste / Miscellaneous: the union of basic tastes named in the
      free-text ``taste`` column (:func:`taste_mentions`). Rows that name no
      basic taste (e.g. only "astringent") are dropped.
    - If ``tasteless`` co-occurs with any taste in a row, ``tasteless`` is
      set to 0 (a tasteless molecule can't also be bitter).

    Args:
        df: Rows from :func:`load_chemtastesdb` and/or :func:`load_bitterdb`
            (stacked with ``pd.concat``). Needs ``source``; ChemTastesDB rows
            also need ``class_taste`` and ``taste``.

    Returns:
        ``(labeled, dropped)``. ``labeled`` is ``df`` plus the label columns
        (int 0/1). ``dropped`` has the removed rows plus a ``drop_reason``.

    Raises:
        ValueError: On an unknown source or ChemTastesDB class, so a new
            category in a future data release can't be silently mislabeled.
    """
    # A fresh index, so rows stacked without ignore_index can't collide.
    df = df.reset_index(drop=True)
    unknown_sources = set(df["source"]) - {"chemtastesdb", "bitterdb"}
    if unknown_sources:
        raise ValueError(f"unknown sources: {sorted(unknown_sources)}")
    is_ctdb = df["source"] == "chemtastesdb"
    classes = df.loc[is_ctdb, "class_taste"]
    unknown = set(classes) - set(_CLASS_TO_LABEL) - _FREE_TEXT_CLASSES
    if unknown:
        raise ValueError(f"unknown ChemTastesDB classes: {sorted(unknown, key=str)}")

    labels = pd.DataFrame(0, index=df.index, columns=LABELS)
    labels.loc[df["source"] == "bitterdb", "bitter"] = 1
    for cls, label in _CLASS_TO_LABEL.items():
        if label is not None:
            labels.loc[is_ctdb & (df["class_taste"] == cls), label] = 1

    free_text = is_ctdb & df["class_taste"].isin(_FREE_TEXT_CLASSES)
    mentions = taste_mentions(df[free_text])[LABELS].astype(int)
    labels.loc[free_text, LABELS] = mentions

    has_taste = labels[LABELS[:-1]].any(axis=1)
    labels.loc[has_taste, "tasteless"] = 0

    labeled = pd.concat([df, labels], axis=1)
    no_taste = free_text & ~labels.any(axis=1)
    labeled, dropped = _split_drops(
        labeled, no_taste, "Multitaste/Miscellaneous naming no basic taste"
    )
    return labeled.reset_index(drop=True), dropped.reset_index(drop=True)


# Labels whose tastant is the ion itself (Na+ for salty, glutamate salts for
# umami), so stripping counterions would remove the very thing being tasted.
_ION_LABELS = ["salty", "umami"]


def _parent_smiles(smiles: str) -> str | None:
    """Largest fragment of ``smiles``, neutralized, as canonical SMILES."""
    mol = _parse(smiles)
    if mol is None:
        return None
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    return Chem.MolToSmiles(mol)


def strip_salts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace each molecule with its neutral parent, except ion-tasting rows.

    For every row *not* labeled salty or umami, ``canonical_smiles`` becomes
    the largest fragment (most atoms), neutralized by RDKit's ``Uncharger``
    (e.g. ``CC(=O)[O-].[Na+]`` -> ``CC(=O)O``). This makes a salt and its
    free acid/base share one key, so dedup merges them and the split can't put
    them on opposite sides. Salty and umami rows keep their full SMILES,
    because the ion is the tastant. Single-fragment charged rows are
    neutralized too, so a salt written without its counterion matches.

    Args:
        df: Output of :func:`assign_labels`. Needs ``canonical_smiles``,
            ``source`` and the :data:`LABELS` columns.

    Returns:
        ``(stripped, log)``. ``stripped`` is ``df`` with updated
        ``canonical_smiles`` and a new ``pre_salt_smiles`` column holding the
        old value. ``log`` has one row per changed molecule: ``row`` (index in
        ``stripped``), ``source``, ``pre_salt_smiles``, ``canonical_smiles``
        and ``change`` (``"largest fragment + neutralized"`` or
        ``"neutralized"``).

    Raises:
        ValueError: If a SMILES can't be parsed (loaders should have dropped
            those, so this means the input skipped a step).
    """
    df = df.reset_index(drop=True)
    keep_ions = df[_ION_LABELS].any(axis=1)
    old = df["canonical_smiles"]
    new = old.where(keep_ions, old.map(_parent_smiles))
    if new.isna().any():
        bad = old[new.isna()].tolist()
        raise ValueError(f"unparsable SMILES: {bad[:5]}")

    stripped = df.assign(pre_salt_smiles=old, canonical_smiles=new)
    changed = stripped[old != new]
    multi = changed["pre_salt_smiles"].str.contains(".", regex=False)
    log = pd.DataFrame(
        {
            "row": changed.index,
            "source": changed["source"],
            "pre_salt_smiles": changed["pre_salt_smiles"],
            "canonical_smiles": changed["canonical_smiles"],
            "change": multi.map(
                {True: "largest fragment + neutralized", False: "neutralized"}
            ),
        }
    ).reset_index(drop=True)
    return stripped, log


def merge_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse rows that share a ``canonical_smiles`` into one molecule.

    Rules (owner decisions, see ``docs/decisions.md``):

    - The multi-hot label vectors of duplicate rows are OR-ed together, so a
      molecule that is sweet in one source and bitter in another is both.
    - After the OR, ``tasteless`` is cleared if any real taste is set (a
      molecule can't be tasteless and bitter at once).

    Args:
        df: Output of :func:`strip_salts` (so salts and their parents already
            share a key). Needs ``canonical_smiles``, ``source``,
            ``pre_salt_smiles`` and the :data:`LABELS` columns.

    Returns:
        ``(merged, log)``. ``merged`` has one row per molecule, sorted by
        ``canonical_smiles``, with the columns ``canonical_smiles``, the
        :data:`LABELS` (int 0/1), ``n_rows`` (input rows merged) and
        ``sources`` (sorted, joined with ``" | "``). ``log`` has one row per
        merged molecule (``n_rows > 1``) with ``canonical_smiles``,
        ``n_rows``, ``sources``, ``input_smiles`` (distinct pre-salt SMILES),
        ``labels`` (labels after the merge) and ``tasteless_cleared``.
    """

    def _join(values: pd.Series) -> str:
        return " | ".join(sorted(values.unique()))

    grouped = df.groupby("canonical_smiles", sort=True)
    labels = grouped[LABELS].max().astype(int)
    tastes = labels[LABELS[:-1]].any(axis=1)
    tasteless_cleared = tastes & (labels["tasteless"] == 1)
    labels.loc[tasteless_cleared, "tasteless"] = 0

    info = grouped.agg(
        n_rows=("source", "size"),
        sources=("source", _join),
        input_smiles=("pre_salt_smiles", _join),
    )
    merged = pd.concat([labels, info[["n_rows", "sources"]]], axis=1).reset_index()

    is_merge = info["n_rows"] > 1
    label_names = labels.apply(
        lambda row: " | ".join(name for name in LABELS if row[name]) or "(none)",
        axis=1,
    )
    log = info.assign(labels=label_names, tasteless_cleared=tasteless_cleared)
    log = log[is_merge].reset_index()
    return merged, log


PROCESSED_DIR = RAW_DIR.parent / "processed"


def build_dataset(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Run the whole Phase 1 cleaning pipeline and write the results to disk.

    Steps: load both sources -> :func:`assign_labels` -> :func:`strip_salts`
    -> :func:`merge_duplicates`. Writes four CSVs to ``out_dir``:
    ``molecules.csv`` (the merged table), ``dropped.csv`` (every dropped row
    with its ``drop_reason``), ``salt_log.csv`` and ``merge_log.csv``.

    Args:
        out_dir: Output folder (created if missing). Defaults to
            ``data/processed/``, which is gitignored.

    Returns:
        The merged table (same as ``molecules.csv``).
    """
    ctdb, ctdb_dropped = load_chemtastesdb()
    bdb, bdb_dropped = load_bitterdb()
    rows = pd.concat([ctdb, bdb], ignore_index=True)
    labeled, label_dropped = assign_labels(rows)
    stripped, salt_log = strip_salts(labeled)
    merged, merge_log = merge_duplicates(stripped)

    dropped = pd.concat([ctdb_dropped, bdb_dropped, label_dropped], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "molecules.csv", index=False)
    dropped.to_csv(out_dir / "dropped.csv", index=False)
    salt_log.to_csv(out_dir / "salt_log.csv", index=False)
    merge_log.to_csv(out_dir / "merge_log.csv", index=False)
    return merged
