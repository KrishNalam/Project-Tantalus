"""Training loop for the GNNs: class-weighted BCE, early stopping on val loss.

Only training and prediction live here; scoring is done by ``evaluate.py``.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tantalus.data import LABELS, PROCESSED_DIR, load_split_table
from tantalus.featurize import smiles_to_graph


def load_graph_table(
    processed_dir: Path = PROCESSED_DIR, splits_file: str = "splits.csv"
) -> tuple[list[Data], pd.DataFrame]:
    """Build one PyG graph per molecule, with its labels attached as ``y``.

    Args:
        processed_dir: Folder written by ``scripts/build_data.py``.
        splits_file: ``splits.csv`` (scaffold) or ``random_splits.csv``.

    Returns:
        ``(graphs, splits)``: ``graphs[i].y`` is a ``(1, len(LABELS))`` float
        tensor, and ``splits`` is aligned row-for-row with ``graphs``.

    Raises:
        ValueError: If a canonical SMILES fails to parse (stale dataset).
    """
    molecules, splits = load_split_table(processed_dir, splits_file)
    labels = torch.tensor(molecules[LABELS].to_numpy(), dtype=torch.float)
    graphs = []
    for i, smiles in enumerate(molecules["canonical_smiles"]):
        graph = smiles_to_graph(smiles)
        if graph is None:
            raise ValueError(f"{smiles!r} failed to parse; rebuild the dataset")
        graph.y = labels[i : i + 1]
        graphs.append(graph)
    return graphs, splits


def pos_weight_from(graphs: list[Data]) -> torch.Tensor:
    """Per-label ``n_negative / n_positive`` for ``BCEWithLogitsLoss``.

    Call this on the **training graphs only**: computing it on val/test would
    let held-out labels shape the loss (leakage). A label with no training
    positives gets weight 1 (there is nothing to up-weight).
    """
    y = torch.cat([g.y for g in graphs])
    n_pos = y.sum(dim=0)
    n_neg = len(y) - n_pos
    return torch.where(n_pos > 0, n_neg / n_pos.clamp(min=1), torch.ones_like(n_pos))


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """One pass over ``loader``; trains if ``optimizer`` is given. Returns mean loss."""
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            loss = loss_fn(model(batch), batch.y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * batch.num_graphs
            count += batch.num_graphs
    return total / count


def train_gnn(
    model: nn.Module,
    train_graphs: list[Data],
    val_graphs: list[Data],
    seed: int,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 200,
    patience: int = 20,
) -> tuple[nn.Module, int]:
    """Train with Adam and keep the weights from the best validation epoch.

    Early stopping: after every epoch the (class-weighted) val loss is
    computed; if it hasn't improved for ``patience`` epochs, training stops
    and the best epoch's weights are restored. Val is only used to pick the
    epoch, never to update weights.

    Args:
        model: An untrained GNN that returns logits (e.g. ``models.gnn.GCN``).
        train_graphs: Graphs with ``y`` used for gradient updates.
        val_graphs: Graphs with ``y`` used only for early stopping.
        seed: Seeds the shuffling order of the training batches.
        lr: Adam learning rate.
        batch_size: Molecules per mini-batch.
        max_epochs: Hard cap on epochs.
        patience: Epochs without val improvement before stopping.

    Returns:
        ``(model, best_epoch)`` with the best-val weights loaded (1-based epoch).
    """
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_from(train_graphs))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_graphs, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, max_epochs + 1):
        _run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = _run_epoch(model, val_loader, loss_fn)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch


@torch.no_grad()
def predict_proba(
    model: nn.Module, graphs: list[Data], batch_size: int = 256
) -> np.ndarray:
    """Return sigmoid probabilities, shape ``(len(graphs), len(LABELS))``."""
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size)
    return torch.cat([torch.sigmoid(model(b)) for b in loader]).numpy()
