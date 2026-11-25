"""Draw an integrated-gradients heatmap for one molecule and one label.

Usage:
    uv run python scripts/explain_molecule.py "<SMILES>" bitter
    uv run python scripts/explain_molecule.py "<SMILES>" sweet --model gin --seed 2

Loads ``data/processed/checkpoints/<model>_seed<k>.pt`` (written by
``scripts/run_gnn.py``), attributes the label's logit to atoms and writes a PNG
to ``data/processed/heatmaps/<model>_seed<k>_<label>_<hash>.png`` (gitignored, like all of ``data/``).
"""

import argparse
import hashlib

import torch

from tantalus.data import LABELS, PROCESSED_DIR
from tantalus.explain import draw_attributions, explain_smiles
from tantalus.models.gnn import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("smiles")
    parser.add_argument("label", choices=LABELS)
    parser.add_argument("--model", default="gin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--out", help="PNG path (default: data/processed/heatmaps/)")
    args = parser.parse_args()

    ckpt = PROCESSED_DIR / "checkpoints" / f"{args.model}_seed{args.seed}.pt"
    model, _ = load_checkpoint(ckpt)
    result = explain_smiles(model, args.smiles, args.label, args.steps)
    if result is None:
        raise SystemExit(f"could not parse SMILES {args.smiles!r}")
    attributions, delta = result

    png = draw_attributions(args.smiles, attributions)
    if args.out:
        out = args.out
    else:
        out_dir = PROCESSED_DIR / "heatmaps"
        out_dir.mkdir(parents=True, exist_ok=True)
        # SMILES contain '/', '\\', '#', so name the file by a short hash instead.
        tag = hashlib.md5(args.smiles.encode()).hexdigest()[:8]
        out = out_dir / f"{args.model}_seed{args.seed}_{args.label}_{tag}.png"
    with open(out, "wb") as f:
        f.write(png)

    total = attributions.sum().item()
    print(
        f"sum of atom scores ({args.label} logit): {total:+.3f}, completeness gap {delta:+.4f}"
    )
    print(f"top atom: {int(torch.argmax(attributions))}; wrote {out}")


if __name__ == "__main__":
    main()
