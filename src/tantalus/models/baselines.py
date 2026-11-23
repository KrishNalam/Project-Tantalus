"""Fingerprint baselines: Morgan fingerprints -> tree models, one per label.

The GNNs in Phase 3 have to beat these on the same scaffold splits. This
module only trains and predicts; scoring lives in the owner's ``evaluate.py``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from tantalus.data import LABELS, PROCESSED_DIR
from tantalus.featurize import smiles_to_fingerprint


def load_fingerprint_table(
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load the merged molecules, their fingerprints, and the saved splits.

    Args:
        processed_dir: Folder holding ``molecules.csv`` and ``splits.csv``
            (written by ``scripts/build_data.py``).

    Returns:
        ``(X, Y, splits)``: ``X`` is an ``(n, 2048)`` uint8 Morgan fingerprint
        matrix, ``Y`` an ``(n, len(LABELS))`` 0/1 label matrix, and ``splits``
        the ``splits.csv`` table aligned row-for-row with ``X`` and ``Y``.

    Raises:
        ValueError: If a molecule has no split row or no fingerprint
            (both would mean the processed files are out of sync).
    """
    molecules = pd.read_csv(processed_dir / "molecules.csv")
    splits = pd.read_csv(processed_dir / "splits.csv")
    # Align on the SMILES key, not on row order (see ADR 2026-11-23).
    splits = molecules[["canonical_smiles"]].merge(
        splits, on="canonical_smiles", how="left", validate="one_to_one"
    )
    if splits.drop(columns="canonical_smiles").isna().any().any():
        raise ValueError("some molecules have no split; rerun scripts/build_data.py")

    fingerprints = [smiles_to_fingerprint(s) for s in molecules["canonical_smiles"]]
    if any(fp is None for fp in fingerprints):
        raise ValueError("a canonical SMILES failed to parse; rebuild the dataset")
    X = np.stack(fingerprints)
    Y = molecules[LABELS].to_numpy(dtype=np.int64)
    return X, Y, splits


def train_random_forest(
    X: np.ndarray, Y: np.ndarray, seed: int, n_estimators: int = 500
) -> list[RandomForestClassifier]:
    """Fit one class-weighted Random Forest per label (binary relevance).

    Args:
        X: Training fingerprints, shape ``(n, n_bits)``.
        Y: Training labels, shape ``(n, n_labels)``, 0/1.
        seed: Random state for the forest (bootstrap and feature sampling).
        n_estimators: Trees per forest.

    Returns:
        One fitted classifier per label column, in ``LABELS`` order.
    """
    models = []
    for j in range(Y.shape[1]):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X, Y[:, j])
        models.append(model)
    return models


def train_xgboost(
    X: np.ndarray, Y: np.ndarray, seed: int, n_estimators: int = 500
) -> list[XGBClassifier | None]:
    """Fit one class-weighted gradient-boosted tree model per label.

    Class imbalance is handled with ``scale_pos_weight = n_negative / n_positive``,
    which up-weights positives the same way ``class_weight="balanced"`` does
    for the Random Forest. Other hyperparameters are XGBoost defaults (no
    tuning, so val/test never influence fitting).

    Args:
        X: Training fingerprints, shape ``(n, n_bits)``.
        Y: Training labels, shape ``(n, n_labels)``, 0/1.
        seed: Random state for XGBoost.
        n_estimators: Boosting rounds.

    Returns:
        One fitted classifier per label column, in ``LABELS`` order. A label
        with only one class in ``Y`` gets ``None`` (XGBoost can't fit it);
        ``predict_proba`` turns that into a constant prediction.
    """
    models: list[XGBClassifier | None] = []
    for j in range(Y.shape[1]):
        n_pos = int(Y[:, j].sum())
        n_neg = len(Y) - n_pos
        if n_pos == 0 or n_neg == 0:
            models.append(None)
            continue
        model = XGBClassifier(
            n_estimators=n_estimators,
            scale_pos_weight=n_neg / n_pos,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X, Y[:, j])
        models.append(model)
    return models


def predict_proba(
    models: list[RandomForestClassifier | XGBClassifier | None], X: np.ndarray
) -> np.ndarray:
    """Return positive-class probabilities, shape ``(n, len(models))``.

    A forest trained on a label with only one class seen gives a constant
    0 or 1 prediction; a ``None`` model (from ``train_xgboost``) gives 0.
    """
    columns = []
    for model in models:
        if model is None:
            columns.append(np.zeros(len(X)))
            continue
        proba = model.predict_proba(X)
        if 1 in model.classes_:
            columns.append(proba[:, list(model.classes_).index(1)])
        else:
            columns.append(np.zeros(len(X)))
    return np.column_stack(columns)
