"""Sanity-check atom attributions against known bitter/sweet/umami compounds.

For each molecule in the panel below we know, from the taste literature, which
part of the structure is held responsible for the taste (its *pharmacophore*,
e.g. saccharin's sulfonyl-imide). This script averages integrated gradients
over the five per-seed checkpoints and asks whether the attributions land on
that part more than chance would predict.

Usage:
    uv run python scripts/sanity_check_attributions.py
    uv run python scripts/sanity_check_attributions.py --model gin --steps 256
    uv run python scripts/sanity_check_attributions.py --model gin_mean_min20

Writes ``data/processed/attribution_sanity.csv`` (``attribution_sanity_<model>.csv``
for any model other than ``gin``, so runs don't overwrite each other) and one mean-attribution
heatmap per molecule under ``data/processed/heatmaps/sanity/`` (gitignored,
like all of ``data/``).

This is a qualitative check, not a metric: the motifs are approximate, the
model may simply not predict the taste for a given molecule (the predicted
probability is reported for that reason), and a miss is a finding about the
model, not a test failure.
"""

import argparse
from collections.abc import Sequence

import pandas as pd
import torch
from rdkit import Chem

from tantalus.data import LABELS, PROCESSED_DIR
from tantalus.explain import average_attributions, draw_attributions
from tantalus.featurize import smiles_to_graph
from tantalus.models.gnn import load_checkpoint
from tantalus.train import predict_proba

# (name, SMILES, label, motif SMARTS patterns, what the motif is)
PANEL = [
    (
        "caffeine",
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
        "bitter",
        ("[CH3][n,N]",),
        "N-methyl groups of the xanthine core",
    ),
    (
        "quinine",
        "COc1ccc2nccc(C(O)C3CC4CCN3CC4C=C)c2c1",
        "bitter",
        ("[NX3]([CX4])([CX4])[CX4]",),
        "basic quinuclidine nitrogen",
    ),
    (
        "denatonium",
        "CC(c1ccccc1)[N+](C)(CC(=O)Nc1c(C)cccc1C)Cc1ccccc1",
        "bitter",
        ("[NX4+]",),
        "quaternary ammonium centre",
    ),
    (
        "saccharin",
        "O=C1NS(=O)(=O)c2ccccc21",
        "sweet",
        ("O=S(=O)[NX3,NX2-]",),
        "sulfonyl-imide glucophore",
    ),
    (
        "acesulfame",
        "CC1=CC(=O)NS(=O)(=O)O1",
        "sweet",
        ("O=S(=O)[NX3,NX2-]",),
        "sulfonyl-imide glucophore",
    ),
    (
        "aspartame",
        "COC(=O)C(Cc1ccccc1)NC(=O)C(N)CC(=O)O",
        "sweet",
        ("[NX3;H2]", "[CX3](=O)[OX2H1,OX1-]"),
        "aspartyl amine + carboxyl head",
    ),
    (
        "sucrose",
        "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O",
        "sweet",
        ("[OX2H]",),
        "hydroxyl groups",
    ),
    (
        "monosodium glutamate",
        "[Na+].[O-]C(=O)C(N)CCC(=O)O",
        "umami",
        ("[CX3](=O)[OX1-,OX2H1]",),
        "the two carboxyl groups",
    ),
]


def motif_atoms(smiles: str, patterns: Sequence[str]) -> set[int]:
    """Atom indices matching any of ``patterns`` in ``smiles`` (union of hits).

    A taste pharmacophore is often several disconnected groups (aspartame's
    free amine and its carboxyl), so the motif is a list of SMARTS rather than
    one connected pattern.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"bad SMILES: {smiles!r}")
    atoms: set[int] = set()
    for smarts in patterns:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise ValueError(f"bad SMARTS: {smarts!r}")
        atoms.update(
            atom for match in mol.GetSubstructMatches(pattern) for atom in match
        )
    return atoms


def main() -> None:
    """Explain every panel molecule with every seed, score the motif overlap."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default="gin",
        help="checkpoint prefix, i.e. a run_gnn.py output name such as gin, gin_mean "
        "or gin_mean_min20 (loads <model>_seed<k>.pt)",
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    checkpoint_dir = PROCESSED_DIR / "checkpoints"
    models = [
        load_checkpoint(checkpoint_dir / f"{args.model}_seed{seed}.pt")[0]
        for seed in args.seeds
    ]
    out_dir = PROCESSED_DIR / "heatmaps" / "sanity"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, smiles, label, patterns, motif in PANEL:
        result = average_attributions(models, smiles, label, args.steps)
        if result is None:
            raise SystemExit(f"could not parse {name}: {smiles!r}")
        mean, std, max_delta = result

        atoms = motif_atoms(smiles, patterns)
        if not atoms:
            raise SystemExit(f"motif {patterns!r} does not match {name}")
        # Rank atoms by attribution and take as many as the motif has, so
        # "precision at |motif|" is directly comparable to the chance rate
        # |motif| / num_atoms (what a random ranking would score).
        top = set(torch.topk(mean, len(atoms)).indices.tolist())
        precision = len(top & atoms) / len(atoms)
        chance = len(atoms) / mean.numel()
        # Second view: share of the positive attribution mass on the motif.
        positive = mean.clamp(min=0)
        mass = positive[list(atoms)].sum().item() / max(positive.sum().item(), 1e-12)

        # An attribution for a label the model does not predict explains a
        # negative answer, so report the probability next to the overlap.
        graph = smiles_to_graph(smiles)
        column = LABELS.index(label)
        proba = sum(predict_proba(m, [graph])[0, column] for m in models) / len(models)

        rows.append(
            {
                "name": name,
                "label": label,
                "proba": round(float(proba), 3),
                "motif": motif,
                "atoms": mean.numel(),
                "motif_atoms": len(atoms),
                "precision_at_k": round(precision, 3),
                "chance": round(chance, 3),
                "positive_mass_on_motif": round(mass, 3),
                "score_sum": round(mean.sum().item(), 3),
                "max_seed_std": round(std.max().item(), 3),
                "max_delta": round(max_delta, 4),
            }
        )
        png = draw_attributions(smiles, mean)
        (
            out_dir / f"{args.model}_mean_{label}_{name.replace(' ', '_')}.png"
        ).write_bytes(png)

    report = pd.DataFrame(rows)
    suffix = "" if args.model == "gin" else f"_{args.model}"
    path = PROCESSED_DIR / f"attribution_sanity{suffix}.csv"
    report.to_csv(path, index=False)
    print(report.to_string(index=False))
    print(f"\nwrote {path} and {len(rows)} heatmaps to {out_dir}")


if __name__ == "__main__":
    main()
