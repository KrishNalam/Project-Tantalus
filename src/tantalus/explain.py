"""Atom-level attributions for the trained GNNs (integrated gradients).

Integrated gradients (Sundararajan et al., 2017) explains one output of a model
by walking in a straight line from a *baseline* input (here: every atom's
feature vector set to zero) to the real input, and adding up the gradients
along the way. Each atom's score is how much its features moved the chosen
label's logit away from the baseline. Bonds and graph structure stay fixed;
only atom features are attributed.
"""

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from tantalus.data import LABELS
from tantalus.featurize import smiles_to_graph


def integrated_gradients(
    model: nn.Module, graph: Data, label: str, steps: int = 128
) -> tuple[torch.Tensor, float]:
    """Integrated-gradients score per atom for one label's logit.

    The path integral is approximated with the midpoint rule: ``steps`` scaled
    copies of the molecule (``alpha * x`` for ``alpha`` in (0, 1)) are run as
    one batch, so a single forward/backward pass gives every gradient.

    Args:
        model: A trained GNN from ``models.gnn`` (put in eval mode here, so
            dropout is off).
        graph: One molecule from ``featurize.smiles_to_graph``.
        label: A name from ``data.LABELS``, e.g. ``"bitter"``.
        steps: Number of points on the path; more is more accurate.

    Returns:
        ``(attributions, delta)``: a ``(num_atoms,)`` tensor in RDKit atom
        order (positive = pushes towards ``label``), and the completeness gap
        ``sum(attributions) - (logit(x) - logit(baseline))``. A ``delta`` near
        zero means ``steps`` was large enough.

    Raises:
        ValueError: If ``label`` is unknown or ``steps`` is less than 1.
    """
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}; expected one of {LABELS}")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    model.eval()
    k = LABELS.index(label)
    x = graph.x
    baseline = torch.zeros_like(x)

    # Batch of `steps` copies of the molecule; only x differs between copies.
    batch = Batch.from_data_list([graph] * steps)
    alphas = (torch.arange(steps, dtype=x.dtype) + 0.5) / steps
    path = baseline + alphas.view(-1, 1, 1) * (x - baseline)  # (steps, atoms, F)
    path.requires_grad_(True)
    batch.x = path.reshape(-1, x.size(1))

    # Attribute the logit, not the probability: sigmoid saturates near 0/1,
    # which would flatten the gradients the method relies on.
    logits = model(batch)[:, k]
    (grads,) = torch.autograd.grad(logits.sum(), path)
    attributions = ((x - baseline) * grads.mean(dim=0)).sum(dim=1)

    with torch.no_grad():
        ends = Batch.from_data_list([graph, graph])
        ends.x = torch.cat([x, baseline])
        end_logits = model(ends)[:, k]
    delta = attributions.sum().item() - (end_logits[0] - end_logits[1]).item()
    return attributions.detach(), delta


def explain_smiles(
    model: nn.Module, smiles: str, label: str, steps: int = 128
) -> tuple[torch.Tensor, float] | None:
    """Featurize ``smiles`` and run :func:`integrated_gradients` on it.

    Returns:
        ``(attributions, delta)`` as in :func:`integrated_gradients`, or
        ``None`` if the SMILES does not parse.
    """
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None
    return integrated_gradients(model, graph, label, steps)
