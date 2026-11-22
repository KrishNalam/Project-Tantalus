# Learning log

## Phase 0 — Setup & featurization

### What is a SMILES string, and why is the same molecule representable multiple ways?

Smiles string - a short text line that describes the chemical structure of a molecule (only uses ASCII charectars)

### Why must SMILES be canonicalized before any dedup, merge, or train/test split?

### Why does `canonicalize` return `None` for invalid input instead of raising?

### In the molecular graph, what are the nodes and edges? Why are hydrogens implicit?

TODO:

### What is atom hybridization (SP / SP2 / SP3), and why might it matter for taste?

TODO:

### What does it mean for a bond to be "conjugated"?

TODO:

### Why encode yes/no properties (aromatic, in ring) as a single 0/1 flag instead of a two-slot one-hot?

TODO:

### How does PyG represent a molecule with no bonds (e.g. `[Na+]`), and why must `edge_attr` still have shape `[0, 7]`?

TODO:

### Why is each bond stored **twice** (both directions) in `edge_index`?

TODO:

### Why are atom/bond features mostly one-hot, and why does each have an "other" slot?

TODO:

### What does a Morgan fingerprint encode? What does "radius 2" mean, and why 2048 bits?

TODO:

### What is Tanimoto similarity, and what does a value of ~0.3 vs ~0.8 tell you?

TODO:

### What is an "activity cliff", and why do sucrose and sucralose illustrate one?

TODO:

### Why are notebooks committed with outputs cleared?

TODO:

### Why does this project use a CPU-only PyTorch build, and how does uv enforce that?

TODO:

### What's the difference between the graph representation (for the GNN) and the fingerprint (for baselines)?

TODO:

### Why pin a dataset to a specific version (Zenodo record) and verify its MD5 checksum, instead of just downloading "the latest"?

TODO:

### What does a `.` in a SMILES string mean (e.g. a drug drawn as a sulfate salt), and why must salts/counter-ions be handled before dedup and splitting?

TODO:

### What do the "NC" and "ND" terms in a CC-BY-NC-ND licence allow and forbid, and what do they mean for publishing a merged dataset or trained model?

TODO:

### ChemTastesDB already ships a "canonical SMILES" column. Why re-canonicalize it with RDKit before dedup or merging?

TODO:

### Why can't we simply strip counter-ions (keep the largest fragment) for every molecule, when some labels are "salty" or "umami"?

TODO:

### What is stereochemistry (the `@`/`@@` in a SMILES), and what do we lose by using stereo-free SMILES? (Hint: enantiomers can taste different.)

TODO:

### What is a Bemis–Murcko scaffold, and why does splitting by scaffold give a more honest test score than a random split?

TODO:

### All acyclic molecules share the empty scaffold `""` (about 18% of our data), and benzene is another ~17%. How should a scaffold split handle a few huge groups like these, and what goes wrong if one lands in the test set?

TODO:

### Our scaffold split shuffles the small scaffold groups with a seed, but always puts the big groups in train. What does "mean ± std over seeds" measure when the split changes with the seed, versus when only the model's random init changes?

TODO:

### All 16 salty molecules are acyclic, so the scaffold split always puts them in train and val/test have 0 salty positives. Why is ROC-AUC undefined for a label with no positives in the test set, and what are the options (drop the label, merge it, report it separately, change the split)?

TODO:

### When the same molecule appears twice with labels "Bitterness" and "Multitaste", is that a label conflict or two compatible labels? How does the answer change between a single-label (softmax) and a multi-label (sigmoid per label) setup?

TODO:

### A salt like sodium acetate is written `CC(=O)[O-].[Na+]`. If we strip the `[Na+]`, the parent is the charged acetate `CC(=O)[O-]`, which doesn't match neutral acetic acid `CC(=O)O` in the dataset. What does "neutralizing" a molecule mean, and why does leaving parents charged hide near-duplicates from dedup?

TODO:

### ChemTastesDB puts 179 molecules in a "Multitaste" class, but its free-text column says which tastes they have (e.g. "Bitter; Umami"). In a multi-label setup, why is it better to expand these into several positive labels than to keep "Multitaste" as its own label? And what is the risk of building labels with keyword matching (e.g. "umami-enhancing" or "Non-sweet/Sweet")?

TODO:

### Each seed of our scaffold split gives a different test set. On provisional labels, umami has between 4 and 46 val positives depending on the seed. Why does a label with very few positives in a split give a noisy (high-variance) ROC-AUC / PR-AUC, and why do we need to check *every* seed, not just seed 0, before trusting the mean ± std?

TODO:
