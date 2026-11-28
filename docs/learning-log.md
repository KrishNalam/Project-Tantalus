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

### GIN updates an atom as `MLP((1 + ε)·h_i + Σ_j h_j)`, a plain **sum** of neighbours. Why can a sum tell apart neighbourhoods that a mean (or GCN's degree-normalised average) cannot? Give a tiny example with atom feature multisets.

TODO:

### What is the Weisfeiler–Lehman (WL) graph isomorphism test, and what does it mean that GIN is "as powerful as" 1-WL? Name a pair of different molecules (or graphs) that 1-WL — and so any GIN — can't distinguish.

TODO:

### GIN uses sum pooling, so bigger molecules get bigger embeddings. Some GIN test probabilities are exactly 0.0 or 1.0 in the saved CSV. Why might sum pooling push logits to extremes, and does that hurt ROC-AUC, calibration, or both?

TODO:

### `GINEConv` builds a neighbour's message as `ReLU(h_j + W·e_ji)`, adding a projection of the bond features to the neighbour's state. What can GINE distinguish that GIN can't (give a pair of molecular fragments), and why does the bond vector have to be projected to the atom-embedding width first?

TODO:

### What is a "message passing neural network" (MPNN) in the Gilmer et al. (2017) sense: message function, aggregation, update, readout? Map GCN, GIN and GINE onto those four parts.

TODO:

### `NNConv` (edge-conditioned convolution) turns each bond's features into a full weight matrix applied to the neighbour. Compare its parameter count and expressiveness with GINE's additive bond term. When would the extra cost be worth it?

TODO:

### What is a PyTorch `state_dict`, and why save `model.state_dict()` plus the model's name instead of `torch.save(model)`? What does `torch.load(..., weights_only=True)` protect against?

TODO:

### Training GIN twice with the same seed on CPU gave different weights once (seed 0), even though all RNGs were seeded. How can multithreaded floating-point sums (e.g. summing neighbour messages) make training non-deterministic, and what does `torch.use_deterministic_algorithms(True)` trade away to fix it?

TODO:

### Integrated gradients attributes `f(x) - f(baseline)` along a straight path from a baseline input. What is the *completeness* axiom, why does it make `delta` a useful correctness check, and why is an all-zero atom-feature vector a questionable "absence of information" baseline for one-hot features?

TODO:

### Why attribute the **logit** rather than the sigmoid probability? What happens to gradients (and so to IG scores) when a model outputs a probability of exactly 1.0?

TODO:

### IG with `steps` points is a Riemann-sum approximation of an integral. Why did `delta` shrink as `steps` grew (caffeine: -2.77 at 16, -0.04 at 256), and why might a ReLU network never reach exactly zero?

TODO:

### Why use a *diverging* colormap (e.g. blue–white–red) centred on zero for attribution heatmaps instead of a sequential one? What can go wrong when each molecule is scaled by its own maximum absolute score, and when would one shared scale across molecules be better?

TODO:

### Averaging attributions over five seeds gave a large across-seed std (up to 16.8 for denatonium, where the mean sum was 21). What does it mean that five models with similar test AUC disagree this much about *which atoms* matter, and what does that imply about publishing a single-seed heatmap?

TODO:

### For sweet molecules the top-attributed atoms were carbons with a near-uniform +4 score each, so the total tracked atom count. Why would sum pooling produce a near-constant per-atom attribution, and how would mean pooling or a size-normalised readout change the IG scores?

TODO:

### "Precision at k" here takes the top-|motif| atoms and compares the overlap to the chance rate |motif| / |atoms|. Why is the chance rate essential to report, and what would be a better statistic for a motif that covers half the molecule?

TODO:

### A taste pharmacophore (saccharin's sulfonyl-imide, the sweetness "AH-B" model) is a claim about what binds a receptor. Why is agreement between a model's attributions and a pharmacophore only weak evidence that the model learned the chemistry, and what else could produce the same agreement?

TODO:

### The API averages probabilities over five seed models and reports their std. Is the across-seed std a good uncertainty estimate? What kind of uncertainty does it capture (and miss) — e.g. why did methane (`C`, nothing like the training data) get salty 0.79?

TODO:

### Why load the model checkpoints once in FastAPI's `lifespan` instead of inside the request handler, and why is the `/predict` handler a plain `def` rather than `async def` when its work is CPU-bound PyTorch?

TODO:

### The endpoint returns the heatmap as base64 inside JSON. What does base64 encoding do, what does it cost, and when would returning `image/png` from a separate endpoint be the better design?

TODO:

### The Gradio page calls `predict.Ensemble` in-process instead of sending requests to the FastAPI endpoint. What are the trade-offs between those two designs (one process vs. two services), and when would you switch the frontend to call the API?

TODO:

### The demo loads the checkpoints lazily with `functools.cache` on the first request, while the API loads them eagerly at startup in `lifespan`. What does each choice cost (first-request latency, startup time, failing fast on a missing checkpoint)?

TODO:

### The README lists "no applicability-domain check" as a limitation. What is a model's applicability domain, how could you flag a SMILES outside it (for example, low nearest-neighbour Tanimoto similarity to the training set), and why does the seed std not catch it?

TODO:

### Training the same GNN with the same seed on 1, 2 and 4 CPU threads gave different weights. Why does the number of threads change a floating-point result, and why does pinning the thread count (rather than only setting seeds) make training reproducible?

TODO:

### The applicability-domain check flags a molecule when its nearest training neighbour has Tanimoto similarity below 0.3. Why does the *seed standard deviation* of the ensemble not already catch out-of-domain molecules, and what kinds of uncertainty are these two numbers measuring?

TODO:

### `domain.py` computes Tanimoto similarity against the whole 3728-molecule table with one matrix product (`ref @ query`), using the identity `|A & B| = A . B` for bit vectors. Why does `|A | B| = |A| + |B| - |A & B|`, and what would change if the fingerprints were counts instead of bits?

TODO:

### Under the random split, test molecules have a median nearest-train Tanimoto similarity of 0.72; under the scaffold split, 0.58. Why does a scaffold split push held-out molecules further from train, and why does that make random-split ROC-AUC an optimistic estimate of performance on new chemistry?

TODO:

### `split_similarity` compares each test molecule against that seed's *train* split only, not the whole dataset. What would go wrong (and which way would the numbers move) if it used the whole `molecules.csv` instead?

TODO:

### Retraining GINE with 1 CPU thread instead of 2 (same seed, same data) moved its pooled sweet test ROC-AUC from 0.81 to 0.84. Why can the number of threads change floating-point results at all, and what does a 0.03 swing from this alone imply about how to compare two architectures?

TODO:

### Switching GIN from sum pooling to mean pooling cut the sweet attribution sums from 30–74 to 7–9 and stopped them tracking atom count. Why does sum pooling make a molecule's score (and its integrated-gradients attributions) grow with size, and what can sum pooling represent that mean pooling cannot?

TODO:

### One mean-pooling GIN seed hit its best validation loss at epoch 3 and stopped there (patience 20), giving a much worse model. How can early stopping lock in an undertrained model, and what are two ways to guard against it?

TODO:

### Skipping the validation pass for the first 20 epochs changed the *training* of every seed, even though validation never updates weights. Iterating a `DataLoader` uses up torch's global random number generator. Why does that change the dropout masks in later training batches, and how would you make a training run's randomness immune to extra evaluation calls?

TODO:

### Retraining GIN on one thread left bitter/umami attributions where they were, but denatonium's integrated-gradients completeness gap grew from 0.45 to 3.5 at the same 128 steps (score sum 33.4). What does the completeness gap measure, why would a different-but-equally-good model need more steps to close it, and does a large gap invalidate the *ranking* of atoms in a heatmap?

TODO:

### On the random split, Random Forest on Morgan fingerprints beats GIN on sweet by about 0.09 ROC-AUC (0.93 vs. 0.84, informal). Why would a fingerprint + tree model profit more than a GNN when test molecules have close analogues in train? What does that say about which split a model comparison should be made on?

TODO:

### The Random Forest's prediction file has ~200 umami test rows with a probability of exactly 0.0 (no tree voted positive), while XGBoost has none. How do ROC-AUC and average precision treat a block of tied scores, and why can a model with many ties look slightly different under the two metrics even when its ranking is otherwise the same?

TODO:
