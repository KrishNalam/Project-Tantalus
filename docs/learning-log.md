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
