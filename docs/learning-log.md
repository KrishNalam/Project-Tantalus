# Learning log

## Phase 0 — Setup & featurization

### What is a SMILES string, and why is the same molecule representable multiple ways?

Smiles string - a short text line that describes the chemical structure of a molecule (only uses ASCII charectars)

### Why must SMILES be canonicalized before any dedup, merge, or train/test split?

TODO:

### Why does `canonicalize` return `None` for invalid input instead of raising?

TODO:

### In the molecular graph, what are the nodes and edges? Why are hydrogens implicit?

TODO:

### Why is each bond stored **twice** (both directions) in `edge_index`?

TODO:

### Why are atom/bond features mostly one-hot, and why does each have an "other" slot?

TODO:

### What does a Morgan fingerprint encode? What does "radius 2" mean, and why 2048 bits?

TODO:

### What is Tanimoto similarity, and what does a value of ~0.3 vs ~0.8 tell you?

TODO:

### Why does this project use a CPU-only PyTorch build, and how does uv enforce that?

TODO:

### What's the difference between the graph representation (for the GNN) and the fingerprint (for baselines)?

TODO:
