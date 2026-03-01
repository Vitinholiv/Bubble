# Bubble

Filter-bubble simulation on social-network graphs.

## Overview

**Bubble** models how information spreads in a social network and whether
"filter bubbles" form or burst under different influence strategies, message
types, and affinity functions.

The core simulation is encapsulated in `BubbleModel`, which builds a
`networkx.Graph` and evolves it through discrete iterations of:

1. **Influencer selection** — pick the most connected nodes per label group.
2. **Influencer update** — blend an external message into their profiles.
3. **User update** — each non-influencer user blends its profile toward
   connected influencers.
4. **Edge update** — edges are created / removed based on pairwise affinity.

## Project Structure

```
Bubble/
├── pyproject.toml            # Packaging & dependencies
├── .gitignore
├── README.md
├── src/
│   └── bubble/
│       ├── __init__.py       # Public API re-exports
│       ├── model.py          # BubbleModel class
│       ├── affinity.py       # Affinity functions (cosine, dot product)
│       ├── selection.py      # Influencer selection strategies
│       ├── messages.py       # Message generators (uniform, unique)
│       ├── metrics.py        # Bubble-burst metrics
│       ├── config.py         # Default config & factory
│       └── visualization.py  # Plotting utilities
├── notebooks/
│   ├── simulator.ipynb       # Main simulation notebook
│   └── teste0_simulacao.ipynb
└── tests/
    ├── __init__.py
    └── test_model.py
```

## Quick Start (Linux)

```bash
# Create & activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install in editable mode with dev extras
pip install -e ".[dev]"

# Run tests
pytest
```

## Quick Start (Windows)
```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install in editable mode with dev extras
pip install -e ".[dev]"

# Run tests
pytest
```

### Usage (Python)

```python
from bubble import BubbleModel, DEFAULT_CONFIG
from bubble.messages import opposite_uniform_message

model = BubbleModel(DEFAULT_CONFIG)
wpn = model.words_per_node

graph = model.run(10, opposite_uniform_message(wpn, 0), oppposite_uniform_message(wpn, 1))
model.plot_edge_counts()
```

### Usage (Notebook)

Open `notebooks/simulator.ipynb` for an interactive walkthrough.

## License

MIT
