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

### ChemTastesDB has 304 "Non-sweetness" molecules. We label them all-zero (not sweet, but also not bitter, umami, etc.). Is "not sweet" the same as "a negative for every taste"? What would label masking in the loss (ignoring unknown labels instead of treating them as 0) change, and why might we want it later?

TODO:

### RDKit's `Uncharger` turns `CC(=O)[O-]` into `CC(=O)O` but leaves a nitro group `[N+](=O)[O-]` alone. Why can't every charge be "neutralized", and what is the difference between a charged salt form and a zwitterion or charge-separated group?

TODO:

### After dedup, 946 of BitterDB's ~1000 molecules also appear in ChemTastesDB. Why would two "independent" sources overlap this much, and why does that make dedup *before* the split essential rather than optional?

TODO:

### We save the split assignment to `splits.csv` keyed by canonical SMILES instead of re-running `scaffold_split` in every training script. What could go wrong if the baseline and the GNN each computed their own split, even with the same seed? And why key on SMILES rather than row indices?

TODO:

### The Random Forest baseline uses `class_weight="balanced"`. What does that do inside the trees (hint: how it changes the impurity of a split), and why does it matter for umami (292 positives) but much less for bitter (1717)? How is it related to class-weighted binary cross-entropy in the GNN?

TODO:

### We train one forest per label ("binary relevance") instead of a single multi-output forest. What information about label co-occurrence (e.g. sweet + bitter) does binary relevance ignore, and why is that acceptable for a baseline?

TODO:

### XGBoost builds trees *sequentially* (boosting), each one fitting the errors of the ones before; a Random Forest builds independent trees on bootstrap samples and averages them (bagging). Which of the two mainly reduces variance and which mainly reduces bias? Why might boosting overfit a small, noisy label like umami more easily than a forest?

TODO:

### For XGBoost we set `scale_pos_weight = n_negative / n_positive` per label. What does that do to the gradient of the log-loss for positive examples, and why does it change the predicted probabilities (calibration) but much less the ROC-AUC ranking?

TODO:

### On the random split, models usually score higher than on the scaffold split, even on the same molecules and labels. Why? What does that gap tell you about how a model will do on a new chemical series, and why do we still never headline the random-split number?

TODO:

### Salty has only 1 positive in the random-split test set for some seeds. What values can ROC-AUC take with exactly one positive, and why is a per-seed AUC like that almost useless? What would you report instead?

TODO:

### The GCN layer updates each atom as a degree-normalised average of its neighbours' (and its own) transformed features: `h_i' = Σ_j (1/√(d_i d_j)) W h_j`. Why normalise by degree at all, and why does stacking 3 layers let an atom "see" atoms up to 3 bonds away? What happens if you stack too many (oversmoothing)?

TODO:

### `GCN.forward` returns logits and the sigmoid lives inside `BCEWithLogitsLoss`. Why is `sigmoid` followed by `log` numerically risky for very large or very small logits, and what does the fused version do instead?

TODO:

### After message passing we have one vector per atom, but we need one prediction per molecule. Compare mean, sum and max pooling: which one can tell a molecule with 2 identical groups from one with 4? Why might that matter for taste?

TODO:

### Early stopping restores the weights from the epoch with the lowest validation loss. Why is that a form of regularisation? And why does it make the **val** scores of the GNN optimistic, so that only **test** scores can be reported?

TODO:

### `pos_weight` for `BCEWithLogitsLoss` is computed from the training split only. What would change, and why would it count as leakage, if it were computed on the full dataset? Why does a label with zero training positives get weight 1 rather than "infinity"?

TODO:

### `model.train()` vs `model.eval()`: which layers in our GCN behave differently between the two modes, and what would go wrong if we predicted on test while still in training mode?

TODO:
