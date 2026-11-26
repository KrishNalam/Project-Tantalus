"""Gradio demo page: type a SMILES, see taste predictions and a heatmap.

Run locally:
    uv run python -m tantalus.demo

then open http://127.0.0.1:7860. The page calls :class:`predict.Ensemble`
in-process (no HTTP hop to ``api.py``), so it shows exactly what the API
returns: mean probability and seed std per label, plus the seed-averaged
integrated-gradients heatmap for one label.
"""

import io
from functools import cache

import gradio as gr
import pandas as pd
from PIL import Image

from tantalus.data import LABELS
from tantalus.predict import HEADLINE_LABELS, Ensemble

AUTO_LABEL = "auto (top label)"
MAX_SMILES_LENGTH = 500  # same cap as the API: IG cost grows with molecule size

EXAMPLES = [
    ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", AUTO_LABEL],  # caffeine
    ["O=C1NS(=O)(=O)c2ccccc21", "sweet"],  # saccharin
    ["NC(CCC(=O)O)C(=O)O", AUTO_LABEL],  # glutamic acid (umami)
    ["[Na+].[Cl-]", AUTO_LABEL],  # table salt
    ["C", AUTO_LABEL],  # methane: out of domain, confidently wrong
]

NOTE = (
    "**How to read this.** Probabilities are averaged over 5 GIN models trained on "
    "different scaffold splits; *seed std* is how much they disagree. Salty and sour "
    "are marked unvalidated: too few examples to score them on the scaffold split. "
    "*Applicability domain* is the Tanimoto similarity to the nearest training "
    "molecule; below 0.3 the prediction is extrapolation and is flagged, because the "
    "model is confident on molecules unlike anything it has seen. Red atoms push "
    "towards the explained label, blue away; "
    "colours are scaled per molecule. Sweet heatmaps tend to track molecule size "
    "rather than chemistry (a known limitation, see docs/decisions.md)."
)


@cache
def get_ensemble() -> Ensemble:
    """Load the five checkpoints once, on the first request."""
    return Ensemble()


def run(smiles: str, label: str) -> tuple[str, pd.DataFrame, Image.Image, str]:
    """Predict one SMILES and return the values the page displays.

    Raises:
        gr.Error: On empty, too long or unparseable input (shown in the page).
    """
    smiles = smiles.strip()
    if not smiles:
        raise gr.Error("Enter a SMILES string.")
    if len(smiles) > MAX_SMILES_LENGTH:
        raise gr.Error(f"SMILES longer than {MAX_SMILES_LENGTH} characters.")
    result = get_ensemble().predict(smiles, None if label == AUTO_LABEL else label)
    if result is None:
        raise gr.Error(f"Could not parse SMILES {smiles!r}.")

    table = pd.DataFrame(
        {
            "label": LABELS,
            "probability": [round(result.mean[k], 3) for k in LABELS],
            "seed std": [round(result.std[k], 3) for k in LABELS],
            "validated": ["yes" if k in HEADLINE_LABELS else "no" for k in LABELS],
        }
    ).sort_values("probability", ascending=False)
    image = Image.open(io.BytesIO(result.attribution_png))
    domain = result.domain
    verdict = (
        "inside the applicability domain"
        if domain.in_domain
        else "⚠️ **outside the applicability domain** — treat these numbers as "
        "extrapolation"
    )
    caption = (
        f"Heatmap explains **{result.explained_label}** "
        f"(p = {result.mean[result.explained_label]:.2f}). "
        f"Integrated-gradients completeness gap: {result.attribution_delta:.2f}.\n\n"
        f"Nearest training molecule: `{domain.nearest_smiles}` "
        f"(Tanimoto {domain.nearest_similarity:.2f}) — {verdict}."
    )
    return result.canonical_smiles, table, image, caption


def build_demo() -> gr.Blocks:
    """Lay out the page: inputs on the left, results on the right."""
    with gr.Blocks(title="Tantalus") as demo:
        gr.Markdown("# Tantalus\nPredict what a molecule tastes like from its SMILES.")
        with gr.Row():
            with gr.Column():
                smiles = gr.Textbox(label="SMILES", placeholder="e.g. CCO")
                label = gr.Dropdown(
                    [AUTO_LABEL, *LABELS], value=AUTO_LABEL, label="Label to explain"
                )
                button = gr.Button("Predict", variant="primary")
                gr.Examples(EXAMPLES, inputs=[smiles, label])
            with gr.Column():
                canonical = gr.Textbox(label="Canonical SMILES", interactive=False)
                table = gr.Dataframe(label="Predictions", interactive=False)
                image = gr.Image(label="Atom attributions", type="pil")
                caption = gr.Markdown()
        gr.Markdown(NOTE)
        outputs = [canonical, table, image, caption]
        button.click(run, inputs=[smiles, label], outputs=outputs)
        smiles.submit(run, inputs=[smiles, label], outputs=outputs)
    return demo


if __name__ == "__main__":
    build_demo().launch()
