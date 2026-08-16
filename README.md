# Figures 2 and 3 Reproduction

This repository contains the code and data required to reproduce Figures 2 and 3 of the paper.

The complete reproduction workflow is contained in a single self-contained Jupyter notebook.

## Contents

- `Figures_2_and_3_Reproduction.ipynb` — code for reproducing Figures 2 and 3.
- `simulation_data/` — precomputed equilibrium results used to generate the published figures.

The `simulation_data/` directory contains one file for each panel:

- `fig2_a.pkl`–`fig2_d.pkl`
- `fig3_a.pkl`–`fig3_d.pkl`

## Reproduction options

The notebook provides two independent reproduction options.

### Option 1 — Use the supplied equilibrium results

This is the recommended and fastest option.

The equilibrium results used in the published figures are loaded from `simulation_data/`, and Figures 2 and 3 are generated directly.

No equilibrium calculations or parameter sweeps are performed.

### Option 2 — Recompute the figures from scratch

This option reproduces the complete computational workflow underlying the figures.

The equilibrium calculations and all eight parameter sweeps are recomputed directly from the model, after which Figures 2 and 3 are generated from the newly computed results.

The supplied files in `simulation_data/` are not required for this option.

Each panel is evaluated on a 600 × 600 parameter grid. Some calculations are performed in parallel and may require several minutes, depending on the available hardware.

## Requirements

The notebook was tested with Python 3.12.7.

Required Python packages:

- NumPy
- Matplotlib
- Seaborn
- SciPy
- Joblib
- tqdm

Install the required packages with:

`pip install numpy matplotlib seaborn scipy joblib tqdm`

## Usage

Open `Figures_2_and_3_Reproduction.ipynb` in Jupyter.

At the beginning of the notebook, choose one of the two reproduction options and follow the corresponding section.

The two options are independent; Option 1 does not need to be run before Option 2.