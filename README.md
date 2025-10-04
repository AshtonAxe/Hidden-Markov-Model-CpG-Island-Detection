# CpG Island Detection using an Eight-State Hidden Markov Model

This project implements an **eight-state Hidden Markov Model (HMM)** to detect **CpG islands** in the human genome, trained on **human chromosome 21** with an existing CpG annotation as ground truth.  

The model estimates transition and emission probabilities using maximum likelihood estimation (MLE) and applies **smoothing** to handle zero probabilities (avoiding `log(0)` issues).  
Final outputs include the trained parameter matrices and predicted CpG island regions.

---

## Overview
- **Model:** Eight-state Hidden Markov Model  
- **Training Data:** Human chromosome 21 DNA sequence and CpG island annotations  
- **Outputs:**  
  - Final transition probability matrix  
  - Final emission probability matrix  
  - Predicted CpG island locations  
- **Key Features:**  
  - Handles zero probabilities via smoothing  
  - Customizable initial state distribution  
  - Fully reproducible workflow with small example data

---

## Quickstart

### Run these commands
```bash
git clone https://github.com/Ashton_Axe/hmm-cpg-island-detection.git
cd hmm-cpg-island-detection
conda create -n hmm_cpg python=3.11
conda activate hmm_cpg
pip install scikit-learn scipy pyranges biopython pyjaspar pysam pyfaidx logomaker anndata torch
python data/chr21.py
python data/chr22.py
python data/cpg_islands.py
python scripts/run_experiment.py
