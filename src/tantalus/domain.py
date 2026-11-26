"""Applicability domain: how close is a query molecule to the training data?

A model only earns trust on molecules that look like the ones it was trained
on. The *applicability domain* is that region of chemical space; a query far
outside it still gets a confident-looking probability, because a sigmoid has
no way to say "I have never seen anything like this".

The check here is the standard cheap one: take the maximum Tanimoto
similarity between the query's Morgan fingerprint and every molecule in the
processed dataset. Tanimoto similarity of two bit vectors is
``|A & B| / |A | B|`` — 1.0 means identical bits, 0.0 means no shared
substructure. If the nearest neighbour is below a threshold, flag the
prediction as out of domain.

Note this is a *data* question, not a model question: the seed spread in
``predict.Ensemble`` measures disagreement between five models that all saw
roughly the same molecules, so they can agree confidently and all be wrong.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tantalus.data import PROCESSED_DIR
from tantalus.featurize import smiles_to_fingerprint

# Below this nearest-neighbour Tanimoto, call the molecule out of domain.
# Calibrated on this dataset: held-out scaffold-test molecules (seed 0) reach a
# median 0.62 similarity to their train set and a 5th percentile of 0.27, so 0.3
# flags ~7% of molecules the model is actually scored on, while methane (0.04)
# and cisplatin (0.11) are flagged. See ADR 2026-11-26.
SIMILARITY_THRESHOLD = 0.3


@dataclass(frozen=True)
class DomainCheck:
    """Result of comparing one molecule against the reference set."""

    nearest_similarity: float
    nearest_smiles: str
    in_domain: bool


def tanimoto_to_reference(fingerprint: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Tanimoto similarity of one fingerprint against each reference row.

    Args:
        fingerprint: Bit vector, shape ``(n_bits,)``.
        reference: Bit matrix, shape ``(n_molecules, n_bits)``.

    Returns:
        Similarities, shape ``(n_molecules,)``. A fingerprint with no bits set
        scores 0 against everything (rather than dividing by zero).
    """
    query = fingerprint.astype(np.float32)
    ref = reference.astype(np.float32)
    # One matrix product gives every intersection count at once: row i of
    # `ref` dotted with `query` counts the bits set in both.
    intersection = ref @ query
    union = ref.sum(axis=1) + query.sum() - intersection
    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )


def nearest_similarities(queries: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Nearest-neighbour Tanimoto similarity of each query row to ``reference``.

    The batched version of :func:`tanimoto_to_reference`: one matrix product
    gives the full ``(n_queries, n_reference)`` intersection matrix, and only
    the row maxima are kept.

    Args:
        queries: Bit matrix, shape ``(n_queries, n_bits)``.
        reference: Bit matrix, shape ``(n_reference, n_bits)``.

    Returns:
        Max similarity per query, shape ``(n_queries,)``.
    """
    q = queries.astype(np.float32)
    ref = reference.astype(np.float32)
    intersection = q @ ref.T
    union = q.sum(axis=1)[:, None] + ref.sum(axis=1)[None, :] - intersection
    similarity = np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )
    return similarity.max(axis=1)


def split_similarity(
    fingerprints: np.ndarray,
    splits: pd.DataFrame,
    threshold: float = SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    """How far each held-out split sits from its own train split.

    For every seed column, each val/test molecule is compared against that
    seed's **train** molecules only (never the whole table, which would count
    the molecule itself as its own neighbour). A scaffold split should push
    held-out molecules further from train than a random split does; that gap
    is the reason random-split scores look better.

    Args:
        fingerprints: One bit vector per row of ``splits``, same order.
        splits: Saved split table (``canonical_smiles`` plus ``seed_<k>``
            columns of ``"train"`` / ``"val"`` / ``"test"``).
        threshold: Out-of-domain cut-off, reported as the fraction below it.

    Returns:
        One row per (seed, split) with ``n``, ``median``, ``p05``, ``p25``,
        ``mean`` similarity and ``frac_below_threshold``.
    """
    rows = []
    for column in [c for c in splits.columns if c.startswith("seed_")]:
        assignment = splits[column].to_numpy()
        train = fingerprints[assignment == "train"]
        for split in ("val", "test"):
            sims = nearest_similarities(fingerprints[assignment == split], train)
            rows.append(
                {
                    "seed": int(column.removeprefix("seed_")),
                    "split": split,
                    "n": len(sims),
                    "median": float(np.median(sims)),
                    "p05": float(np.percentile(sims, 5)),
                    "p25": float(np.percentile(sims, 25)),
                    "mean": float(sims.mean()),
                    "frac_below_threshold": float((sims < threshold).mean()),
                }
            )
    return pd.DataFrame(rows)


class ApplicabilityDomain:
    """Nearest-neighbour similarity check against the processed dataset."""

    def __init__(self, smiles: list[str], fingerprints: np.ndarray) -> None:
        self.smiles = smiles
        self.fingerprints = fingerprints

    @classmethod
    def from_dataset(
        cls, processed_dir: Path = PROCESSED_DIR, molecules_file: str = "molecules.csv"
    ) -> "ApplicabilityDomain":
        """Build the reference set from ``molecules.csv``.

        The reference is the whole merged table, not one seed's train split:
        the demo ensembles five models whose scaffold splits differ, so a
        molecule held out by one seed was trained on by the others. The union
        is the chemical space the ensemble as a whole has seen.

        Raises:
            FileNotFoundError: If the processed table is missing (run
                ``scripts/build_data.py`` first).
        """
        table = pd.read_csv(processed_dir / molecules_file)
        kept: list[str] = []
        rows: list[np.ndarray] = []
        for smiles in table["canonical_smiles"]:
            fingerprint = smiles_to_fingerprint(smiles)
            if fingerprint is None:  # already canonicalized, so this is unexpected
                continue
            kept.append(smiles)
            rows.append(fingerprint)
        return cls(kept, np.vstack(rows))

    def check(
        self, smiles: str, threshold: float = SIMILARITY_THRESHOLD
    ) -> DomainCheck | None:
        """Nearest reference molecule and whether the query is in domain.

        Returns ``None`` if the SMILES does not parse.
        """
        fingerprint = smiles_to_fingerprint(smiles)
        if fingerprint is None:
            return None
        similarities = tanimoto_to_reference(fingerprint, self.fingerprints)
        best = int(np.argmax(similarities))
        similarity = float(similarities[best])
        return DomainCheck(similarity, self.smiles[best], similarity >= threshold)
