"""Graph neural networks for multi-label taste prediction.

Every model maps a batch of molecular graphs (from ``featurize.smiles_to_graph``)
to one raw score (logit) per label. The sigmoid is applied by the loss
(``BCEWithLogitsLoss``) during training and by the caller at prediction time,
never inside the model; that is the numerically stable way to do it.
"""

from itertools import pairwise
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GINEConv,
    global_add_pool,
    global_mean_pool,
)

from tantalus.data import LABELS
from tantalus.featurize import ATOM_FEATURE_DIM, BOND_FEATURE_DIM

# Graph readouts: atom states (num_atoms, d) + batch index -> (num_graphs, d).
POOLINGS = {"sum": global_add_pool, "mean": global_mean_pool}


class GCN(nn.Module):
    """Graph Convolutional Network (Kipf & Welling, 2017) with a graph-level head.

    Architecture: ``num_layers`` x (GCNConv -> ReLU -> dropout) on the atoms,
    mean pooling over each molecule's atoms, then a two-layer MLP that outputs
    one logit per label. GCNConv ignores bond features (``edge_attr``);
    ``GINE`` is the model that uses them.
    """

    def __init__(
        self,
        in_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_labels: int = len(LABELS),
        dropout: float = 0.2,
    ) -> None:
        """Build the layers.

        Args:
            in_dim: Atom feature size (``featurize.ATOM_FEATURE_DIM``).
            hidden_dim: Width of every message-passing layer and the MLP.
            num_layers: Message-passing layers; each one lets an atom see one
                bond further, so 3 layers cover a radius-3 neighbourhood.
            num_labels: Output logits, one per taste label.
            dropout: Dropout probability after each layer (training only).

        Raises:
            ValueError: If ``num_layers`` is less than 1.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            GCNConv(d_in, d_out) for d_in, d_out in pairwise(dims)
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """Return logits of shape ``(num_graphs, num_labels)``.

        Args:
            batch: A PyG ``Batch`` with ``x``, ``edge_index`` and ``batch``
                (the atom -> molecule index that pooling needs).
        """
        h = batch.x
        for conv in self.convs:
            h = self.dropout(torch.relu(conv(h, batch.edge_index)))
        pooled = global_mean_pool(h, batch.batch)
        return self.head(pooled)


class GIN(nn.Module):
    """Graph Isomorphism Network (Xu et al., 2019) with a graph-level head.

    Architecture: ``num_layers`` x (GINConv -> ReLU -> dropout), sum pooling
    over each molecule's atoms, then the same two-layer MLP head as ``GCN``.
    Each GINConv updates an atom as ``MLP(h_i + sum_j h_j)``: an unweighted
    **sum** of neighbours (no degree normalisation) passed through a small MLP,
    which is what makes GIN as expressive as the Weisfeiler-Lehman test.
    ``eps`` is fixed at 0 ("GIN-0"). Width, depth, dropout and head match
    ``GCN`` so the two models differ only in aggregation and pooling. Bond
    features are ignored, as in ``GCN``. ``pooling="mean"`` swaps the readout
    for an average (see ``GINMean``).
    """

    def __init__(
        self,
        in_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_labels: int = len(LABELS),
        dropout: float = 0.2,
        pooling: str = "sum",
    ) -> None:
        """Build the layers.

        Args:
            in_dim: Atom feature size (``featurize.ATOM_FEATURE_DIM``).
            hidden_dim: Width of every message-passing layer and the MLP.
            num_layers: Message-passing layers (radius of each atom's view).
            num_labels: Output logits, one per taste label.
            dropout: Dropout probability after each layer (training only).
            pooling: Graph readout, ``"sum"`` (GIN paper) or ``"mean"``.

        Raises:
            ValueError: If ``num_layers`` is less than 1 or ``pooling`` is
                unknown.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {sorted(POOLINGS)}")
        self.pool = POOLINGS[pooling]
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            GINConv(
                nn.Sequential(
                    nn.Linear(d_in, d_out), nn.ReLU(), nn.Linear(d_out, d_out)
                )
            )
            for d_in, d_out in pairwise(dims)
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """Return logits of shape ``(num_graphs, num_labels)``.

        Args:
            batch: A PyG ``Batch`` with ``x``, ``edge_index`` and ``batch``.
        """
        h = batch.x
        for conv in self.convs:
            h = self.dropout(torch.relu(conv(h, batch.edge_index)))
        pooled = self.pool(h, batch.batch)
        return self.head(pooled)


class GINMean(GIN):
    """``GIN`` with mean pooling instead of sum pooling: one change, nothing else.

    A controlled experiment, not a new architecture. Sum pooling makes the
    molecule vector grow with atom count, and the Phase 4 sanity check found
    sweet attributions that track molecule size (ADR 2026-11-25). Averaging
    removes size from the readout, so if the artefact disappears here, the
    readout was the cause. The trade-off: mean pooling can no longer count
    (a molecule and its doubled copy pool to the same vector).
    """

    def __init__(self, **kwargs) -> None:
        """Build a ``GIN`` with ``pooling="mean"``; other kwargs pass through."""
        super().__init__(pooling="mean", **kwargs)


class GINE(GIN):
    """GIN with bond features (Hu et al., 2020): the project's bond-aware MPNN.

    Identical to ``GIN`` (depth, width, MLPs, sum pooling, head) except that
    each layer uses ``GINEConv``, whose message from neighbour ``j`` to atom
    ``i`` is ``ReLU(h_j + W e_ji)``: the neighbour's state shifted by a learned
    linear projection of the bond's features (type, conjugated, in ring). So
    an O bonded by a double bond sends a different message than an O bonded by
    a single bond, which plain GIN cannot tell apart. The atom update is then
    ``MLP(h_i + sum_j message_j)`` with ``eps`` fixed at 0, as in ``GIN``.
    """

    def __init__(
        self,
        in_dim: int = ATOM_FEATURE_DIM,
        edge_dim: int = BOND_FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_labels: int = len(LABELS),
        dropout: float = 0.2,
    ) -> None:
        """Build the layers.

        Args:
            in_dim: Atom feature size (``featurize.ATOM_FEATURE_DIM``).
            edge_dim: Bond feature size (``featurize.BOND_FEATURE_DIM``).
            hidden_dim: Width of every message-passing layer and the MLP.
            num_layers: Message-passing layers (radius of each atom's view).
            num_labels: Output logits, one per taste label.
            dropout: Dropout probability after each layer (training only).

        Raises:
            ValueError: If ``num_layers`` is less than 1.
        """
        super().__init__(in_dim, hidden_dim, num_layers, num_labels, dropout)
        dims = [in_dim] + [hidden_dim] * num_layers
        # Replace GIN's convs; each layer gets its own bond projection
        # (edge_dim -> that layer's input width, done inside GINEConv).
        self.convs = nn.ModuleList(
            GINEConv(
                nn.Sequential(
                    nn.Linear(d_in, d_out), nn.ReLU(), nn.Linear(d_out, d_out)
                ),
                edge_dim=edge_dim,
            )
            for d_in, d_out in pairwise(dims)
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """Return logits of shape ``(num_graphs, num_labels)``.

        Args:
            batch: A PyG ``Batch`` with ``x``, ``edge_index``, ``edge_attr``
                (row-aligned with ``edge_index``) and ``batch``.
        """
        h = batch.x
        for conv in self.convs:
            h = self.dropout(torch.relu(conv(h, batch.edge_index, batch.edge_attr)))
        pooled = self.pool(h, batch.batch)
        return self.head(pooled)


MODELS: dict[str, type[nn.Module]] = {
    "gcn": GCN,
    "gin": GIN,
    "gin_mean": GINMean,
    "gine": GINE,
}


def load_checkpoint(path: Path) -> tuple[nn.Module, dict]:
    """Rebuild a trained GNN saved by ``scripts/run_gnn.py``.

    The checkpoint holds the model name, its ``state_dict`` and run metadata
    (seed, split, best epoch). Models are rebuilt with their default
    hyperparameters, which is how ``run_gnn.py`` trains them.

    Args:
        path: A ``.pt`` file from ``data/processed/checkpoints/``.

    Returns:
        ``(model, metadata)``: the model in eval mode with trained weights,
        and the checkpoint's metadata (everything except the weights).
    """
    # weights_only=True: only tensors and plain Python types are unpickled,
    # so a tampered checkpoint can't run arbitrary code on load.
    checkpoint = torch.load(path, weights_only=True)
    model = MODELS[checkpoint["model"]]()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    return model, metadata
