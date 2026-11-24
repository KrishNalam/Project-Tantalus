"""Graph neural networks for multi-label taste prediction.

Every model maps a batch of molecular graphs (from ``featurize.smiles_to_graph``)
to one raw score (logit) per label. The sigmoid is applied by the loss
(``BCEWithLogitsLoss``) during training and by the caller at prediction time,
never inside the model; that is the numerically stable way to do it.
"""

from itertools import pairwise

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, global_mean_pool

from tantalus.data import LABELS
from tantalus.featurize import ATOM_FEATURE_DIM


class GCN(nn.Module):
    """Graph Convolutional Network (Kipf & Welling, 2017) with a graph-level head.

    Architecture: ``num_layers`` x (GCNConv -> ReLU -> dropout) on the atoms,
    mean pooling over each molecule's atoms, then a two-layer MLP that outputs
    one logit per label. GCNConv ignores bond features (``edge_attr``); the
    later MPNN is the model that uses them.
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
    features are ignored, as in ``GCN``.
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
            num_layers: Message-passing layers (radius of each atom's view).
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
        pooled = global_add_pool(h, batch.batch)
        return self.head(pooled)
