# Astro Link Forecasting

This repository accompanies the paper:

**Predicting New Concept–Object Associations in Astronomy by Mining the Literature**

It provides the full experimental pipeline used to construct a large-scale literature-derived concept–object graph and to forecast future associations under a temporal evaluation protocol.

The core scientific task is:

> Given a cutoff year *T*, train on all concept–object associations observed up to *T*,  
> and evaluate how well different methods rank objects whose association first appears after *T*.

The default configuration produces both:
- Results **with inference-time concept smoothing** (main results in the paper), and  
- Results **without smoothing**.

---

# 1. Repository Overview

This repository implements the complete forecasting workflow described in the paper:

1. Construction of temporal train/test splits
2. Concept–object graph assembly from literature-derived data
3. Concept embedding neighbor construction for smoothing and embedding-based baselines
4. Training and evaluation of forecasting methods
5. Stratified metric aggregation

---

# 2. Required External Data

## Concept Data (AstroMLab 5)

Paper–concept associations and concept embeddings are sourced from:

**Ting et al. (2025), AstroMLab 5: Structured Summaries and Concept Extraction for 400,000 Astrophysics Papers**

Required files (place in `data/`):

- `concepts_embeddings.npz`
- `concepts_vocabulary.csv`
- `papers_concepts_mapping.csv`
- `papers_year_mapping.csv`

This repository does not redistribute AstroMLab 5 data.

---

## Object Extraction Data (This Work)

This repository expects LLM-extracted paper–object mention data:

- `paper_object_edges_llm_mentions.jsonl`
- SIMBAD name resolution cache (`simbad_name_resolution_cache_*.jsonl`)

These files contain:

- extracted object mentions
- semantic role labels
- study-mode labels
- SIMBAD-resolved canonical object IDs

All concept–object edges and weights used in experiments are generated from mention-level JSONL inputs and configuration-defined weighting rules. No precomputed weighted graph is required.

---

# 3. Installation

### Python Version

Tested with Python 3.10+.

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Configuration

All experiments are controlled via:

```
config/table1.yaml
```

---

## Configuration Reference

### paths

Defines:

- input data directory
- output directory
- filenames for required inputs

All object–concept edges are constructed dynamically from:

```
paper_object_edges_llm_mentions.jsonl
```

---

### weights

Edge weights are computed as:

w(c,o) = log(1 + Σ_m ρ_r(m) × γ_σ(m))

where:
- ρ_r(m) is the role weight
- γ_σ(m) is the study-mode multiplier

Users may modify:

```yaml
weights:
  role_weight:
  study_mode_mult:
```

and regenerate the graph.

---

### edge_configs

Controls:

- role filtering
- study filtering
- weighting scheme
- per-paper normalization
- caching behavior

---

### cutoffs

Example:

```yaml
cutoffs: [2017, 2019, 2021, 2023]
```

---

### min_train_pos

Minimum number of training associations required for a concept to be evaluated.

---

### knn

Neighborhood sizes and similarity parameters.

---

### smoothing

Inference-time concept smoothing parameters:

\[
s_{smooth}(c,o) = (1-\lambda)s(c,o) + \lambda \sum_{c'} S_{c,c'} s(c',o)
\]

---

### als

Implicit ALS hyperparameters:

- latent factors
- regularization
- iterations
- alpha
- seeds

To reproduce full paper averages, use multiple seeds.

---

### methods

Enable or disable forecasting methods.

---

### output

Defines:

- output subdirectory
- which strata to report
- CSV export path

---

# 5. Sample Workflow

From the repository root:

```bash
bash scripts/reproduce_table1.sh config/table1.yaml
```

This executes:

1. `prepare_cutoff.py`
2. `smoothing.py`
3. `train_eval.py`

Outputs are written to:

```
OUT_DIR/
  table1/
    _global/
    T=2017/
    T=2019/
    T=2021/
    T=2023/
    eval_stratified_results.csv
    table1.tex
```

---

# 6. Reproducibility

- Graph construction is fully reproducible from mention-level JSONL inputs.
- Weighting is config-defined.
- Temporal splits follow strict Mode B semantics.
- No pre-aggregated graph artifacts are required.

---

# 7. Citation

(placeholder — add citation after review period)
