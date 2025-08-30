# K-means Community Detection Analysis

This project implements community detection using iterative k-means clustering followed by co-occurrence analysis on the multishapes dataset.

## Setup with uv

1. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

## Running the Analysis

```bash
python kmeans_analysis.py
```

This will:
- Load the multishapes.csv dataset
- Perform K-means clustering
- Generate visualization plots
- Display cluster analysis results