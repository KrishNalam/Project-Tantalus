"""Seed-ensemble inference: SMILES in, taste probabilities and a heatmap out.

This is the framework-free core of the Phase 5 demo. ``api.py`` wraps it in
HTTP; a notebook or a Gradio page can call it directly.

The five per-seed checkpoints are used together: probabilities are averaged
and their spread reported, and attributions are averaged with
``explain.average_attributions``. Phase 4 showed single-seed attributions
disagree a lot, so the demo never shows one seed on its own.

Every prediction also carries an applicability-domain check
(:mod:`tantalus.domain`): the seed spread says how much the models disagree,
which is not the same as whether the molecule resembles anything they were
trained on.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Batch

from tantalus.data import LABELS, PROCESSED_DIR
from tantalus.domain import ApplicabilityDomain, DomainCheck
from tantalus.explain import average_attributions, draw_attributions
from tantalus.featurize import canonicalize, smiles_to_graph
from tantalus.models.gnn import load_checkpoint

CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints"

# Salty and sour have too few scaffold-test positives to be scored there
# (CLAUDE.md, Phase 1 decisions), so their outputs are flagged as unvalidated.
HEADLINE_LABELS = ["sweet", "bitter", "umami", "tasteless"]


@dataclass
class Prediction:
    """Ensemble output for one molecule."""

    canonical_smiles: str
    mean: dict[str, float]
    std: dict[str, float]
    explained_label: str
    attribution_png: bytes
    attribution_delta: float
    domain: DomainCheck


class Ensemble:
    """The per-seed checkpoints of one architecture, used as one predictor."""

    def __init__(
        self,
        model: str = "gin",
        seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
        checkpoint_dir: Path = CHECKPOINT_DIR,
        domain: ApplicabilityDomain | None = None,
    ) -> None:
        """Load ``<checkpoint_dir>/<model>_seed<k>.pt`` for every seed.

        Also builds the applicability-domain reference set from
        ``molecules.csv`` (pass ``domain`` to reuse one), so every prediction
        can report how close the query is to the training data.

        Raises:
            FileNotFoundError: If a checkpoint or ``molecules.csv`` is missing
                (run ``scripts/run_gnn.py <model>`` / ``scripts/build_data.py``
                first).
        """
        self.model_name = model
        self.domain = domain or ApplicabilityDomain.from_dataset()
        self.models: list[nn.Module] = [
            load_checkpoint(checkpoint_dir / f"{model}_seed{s}.pt")[0] for s in seeds
        ]

    @torch.no_grad()
    def probabilities(self, smiles: str) -> torch.Tensor | None:
        """Sigmoid outputs, shape ``(num_models, len(LABELS))``; ``None`` if invalid."""
        graph = smiles_to_graph(smiles)
        if graph is None:
            return None
        batch = Batch.from_data_list([graph])
        return torch.cat([torch.sigmoid(m(batch)) for m in self.models])

    def predict(
        self, smiles: str, label: str | None = None, steps: int = 128
    ) -> Prediction | None:
        """Predict all labels and explain one of them.

        Args:
            smiles: Any valid SMILES; it is canonicalized first, as in training,
                and the canonical string is used for both the graph and the
                drawing so atom order lines up.
            label: Label to explain. Default: the label with the highest mean
                probability (salty/sour included; callers flag them via
                :data:`HEADLINE_LABELS`).
            steps: Integrated-gradients path steps per model.

        Returns:
            A :class:`Prediction`, or ``None`` if the SMILES does not parse.

        Raises:
            ValueError: If ``label`` is not in ``data.LABELS``.
        """
        if label is not None and label not in LABELS:
            raise ValueError(f"unknown label {label!r}; expected one of {LABELS}")
        canonical = canonicalize(smiles)
        if canonical is None:
            return None
        probs = self.probabilities(canonical)
        if probs is None:
            return None
        mean = dict(zip(LABELS, probs.mean(dim=0).tolist(), strict=True))
        std = dict(zip(LABELS, probs.std(dim=0, unbiased=False).tolist(), strict=True))
        if label is None:
            label = max(LABELS, key=mean.__getitem__)

        scores, _, max_delta = average_attributions(
            self.models, canonical, label, steps
        )
        png = draw_attributions(canonical, scores)
        check = self.domain.check(canonical)
        if check is None:  # unreachable: the SMILES already parsed above
            raise RuntimeError(f"fingerprint failed for canonical {canonical!r}")
        return Prediction(canonical, mean, std, label, png, max_delta, check)
