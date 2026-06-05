# Unsupervised Learning Project: Hotel Booking Demand (Final)

**Course:** Unsupervised Learning 2025/2026 | NOVA FCT 
**Team:**  

* Afonso Simões - 73204  
* José Moutinho - 73129  
* Luís Nunes - 73216  

## Project Overview

This repository implements the final pipeline used in the course project to discover booking profiles in the Hotel Booking Demand dataset. It is a reproducible, end-to-end pipeline that enforces leakage control, builds configurable preprocessing representations, evaluates multiple clustering algorithms, and logs metrics and stability diagnostics for downstream analysis.

## Repository layout (final)

- `data/raw/` - add the raw `hotel_bookings_course_release_v1.csv` here (ignored by git).
- `data/` - course release artifacts, including `column_roles.csv` and `SHA256SUMS.txt` for integrity checks.
- `src/` - implementation modules:
	- `data_loader.py` - loads raw data and prints SHA-256 hash.
	- `preprocessing.py` - builds the `ColumnTransformer` (log transforms, `StandardScaler`/`RobustScaler`, OHE).
	- `modeling.py` - iK-Means implementation and helpers.
	- `evaluation.py` - K-Means, GMM training loops, internal metrics and stability routines.
	- `tests/` - quick validation scripts used during development.
- `notebooks/` - EDA and analysis notebooks.
- `tables/` - generated outputs; key files include:
	- `tables/experiments.csv` - main experiment log (methods, parameters, metrics, stability stats).
	- `tables/profiling_base_data.csv` - outcome and profile variables saved separately for post-hoc analysis.
	- other artifacts: `cluster_metrics_by_k_summary.csv`, `bootstrap_stability_*.csv`, `X_scaled_matrix_*.csv`, `final_feature_space.csv`.
- `figures/` - generated figures and plots.
- `run_all.py` - orchestrates the full pipeline and writes results to `tables/`.



## Setting up the Environment

To avoid any version conflicts and ensure reproducibility, please set up the environment using Conda:
```bash
conda env create -f environment.yml
conda activate unsupervised_env
```

## Quick start - reproduce the final run

1. Place `hotel_bookings_course_release_v1.csv` in `data/raw/`.

2. Run the full pipeline (may take substantial time depending on machine):

```bash
python run_all.py
```

3. For a quick smoke check use fast mode:

```bash
python run_all.py --fast
```

The pipeline prints the SHA-256 hash of the raw file on load and saves results under `tables/` (notably `tables/experiments.csv`).

## What the pipeline does (high level)

- Preprocesses the raw dataset with configurable scaling and categorical encoding. Low-frequency countries are grouped into `country_grouped` and several derived numerical features are created (e.g., `total_nights`, `party_size`).
- Constructs multiple representations by toggling scaler (`standard` / `robust`) and whether `hotel` is included.
- Runs K-Means across a grid of `K` values and random seeds, evaluates internal metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) and estimates stability via bootstrap ARI.
- Runs Gaussian Mixture Models (GMM) and reports seed-based ARI stability.
- Runs iK-Means anomalous-cluster extraction to produce deterministic initial centroids and evaluates the resulting clustering.

## Outputs and interpretation

- `tables/experiments.csv` contains one row per method/run including `representation_id`, `method`, `k`, `seed`, `silhouette`, `calinski_harabasz`, `davies_bouldin`, runtime and stability statistics (`mean_ARI`, `std_ARI`, `min_ARI`, `max_ARI`).
- `tables/profiling_base_data.csv` stores `adr`, `is_canceled` and `reservation_status` for post-clustering profiling - these are not used as clustering features to prevent leakage.

Use these outputs and the notebooks in `notebooks/` to reproduce the figures and analyses used in the course deliverables.

## Testing and quick validation

- Run unit/validation scripts with `pytest` (or execute specific test scripts):

```bash
pytest -q
# or run a single script
python src/tests/test_evaluation.py
```

## Notes, design choices and reproducibility

- Stability: K-Means stability is estimated via bootstrap ARI across resampled overlapping samples; GMM stability is estimated by pairwise ARI across seeds. iK-Means is deterministic (given parameters) and used as a specialized initializer.
- Feature choices: the pipeline compares `StandardScaler` and `RobustScaler`, and experiments toggling inclusion of the `hotel` identifier to assess whether hotel-specific effects dominate clustering.
- Reproducibility: all experiments are logged (parameters + seed) to `tables/experiments.csv` and the raw data hash is printed at load time for governance.

---

Acknowledgements: developed for the Unsupervised Learning course at NOVA FCT.