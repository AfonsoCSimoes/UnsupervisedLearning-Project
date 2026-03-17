# Unsupervised Learning Project: Hotel Booking Demand

**Course:** Unsupervised Learning 2025/2026 | NOVA FCT

**Team:** 
* Afonso Simões - 73204  
* José Moutinho - [Inserir Número]  
* Luís Nunes - [Inserir Número]  

## 📌 Overview
This repository contains the code and methodology for a reproducible clustering study on the real-world Hotel Booking Demand dataset. The goal is to design an end-to-end data mining pipeline to discover coherent "booking profiles" without the use of class labels, while strictly enforcing leakage control (excluding outcome/post-event variables) and proper feature governance.

## 📂 Repository Structure
Conforming to the project's reproducibility requirements, the repository is organized as follows:

* `data/raw/`: Directory for the raw dataset. **(Ignored by git - do not commit data here)**.
* `data/metadata/`: Contains course release artifacts (`column_roles.csv`, `SHA256SUMS.txt` `DATASET_MANIFEST.yml`, and subsample indices) to ensure data provenance and methodological governance.
* `src/`: Source code for data preprocessing, modeling, evaluation, and robustness analysis.
* `figures/` & `tables/`: Output directories where the pipeline automatically saves all visual and tabular results.
* `experiments.csv`: Machine-readable log detailing method, parameters, seeds, metrics, and diagnostics.
* `environment.yml`: Conda environment specification.
* `run_all.py`: Single-entry point script to regenerate all results end-to-end.

## 🚀 Getting Started & Dataset Setup

**⚠️ CRITICAL: The raw dataset is NOT included in this repository.** To run the project locally, you must follow these exact steps to ensure the pipeline works:

1. **Obtain the Dataset:** Download `hotel_bookings_course_release_v1.csv` from the course platform.
2. **Place the File:** Move the file directly into the `data/raw/` folder.  
*(Note: This is strictly required so it is possible to run the project locally without pushing a giant `.csv` file to GitHub.)*
3. **Verify Integrity:** You can use the provided `SHA256SUMS.txt` to verify your local copy matches the authoritative "course release v1".

## 🛠️ Environment Setup
To ensure reproducibility, install the exact dependencies used by our team:

```bash
conda env create -f environment.yml
conda activate unsupervised_env