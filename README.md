# Unsupervised Learning Project: Hotel Booking Demand

**Course:** Unsupervised Learning 2025/2026 | NOVA FCT  
**Team:**  

* Afonso Simões - 73204  
* José Moutinho - 73129  
* Luís Nunes - 73216  

## Project Overview

Welcome to our project repository for the Hotel Booking Demand dataset! We built this project to explore and implement an end-to-end data mining pipeline capable of discovering coherent "booking profiles." Rather than relying on class labels, our approach uses unsupervised clustering techniques. A major focus of our work has been strictly enforcing leakage control—making sure outcome and post-event variables are excluded from the clustering space—and establishing strong data governance.

## Repository Layout

We've organized our workspace carefully to ensure everything is reproducible and easy to navigate:

* `data/raw/`: The directory where the raw dataset should be placed. *(Note: This folder is ignored by git so we don't commit large data files.)*
* `data/`: Contains essential course release artifacts (like `column_roles.csv`, `SHA256SUMS.txt`, and `DATASET_MANIFEST.yml`) used to verify our data's integrity.
* `src/`: Holds all our core Python modules for preprocessing, modeling, evaluation, and robustness testing.
* `notebooks/`: Jupyter notebooks that walk through our Exploratory Data Analysis (EDA), feature engineering, and pipeline validation.
* `figures/` & `tables/`: Auto-generated directories where our scripts save plots and tabular data outputs.
* `experiments.csv`: An automated log that records all method configurations, parameters, random seeds, evaluation metrics, and runtimes.
* `environment.yml`: The Conda environment specification to ensure dependencies match perfectly.
* `run_all.py`: The single-entry point script to execute the entire pipeline from start to finish.

## Milestone 2 Progress

As of the April 30th checkpoint, we have successfully implemented:

1. **Data Governance & EDA:** We established strict dataset hash verification, mapped out missingness and outliers, and formally documented feature roles (ensuring leakage variables are strictly profiled post-clustering).
2. **Preprocessing Pipeline:** We built a robust `scikit-learn` ColumnTransformer to handle missing values, apply numerical scaling (testing both `StandardScaler` and `RobustScaler`), and execute full one-hot encoding on our categorical features.
3. **Baseline Modeling:** We integrated standard K-Means alongside an implementation of iK-Means, testing over a defined grid of $K$ values within our unified Euclidean representation matrix.
4. **Experiment Tracking:** The pipeline automatically evaluates stability and internal metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) and logs them directly to `experiments.csv`.

## Setting up the Dataset

To run the pipeline locally, you will need to add the dataset manually:

1. Download the `hotel_bookings_course_release_v1.csv` file from the course platform.
2. Place the file into the `data/raw/` folder in this project directory.
3. Our scripts will automatically check this file against the `SHA256SUMS.txt` hash to ensure your local copy is valid before execution.

## Setting up the Environment

To avoid any version conflicts and ensure reproducibility, please set up the environment using Conda:
```bash
conda env create -f environment.yml
conda activate unsupervised_env