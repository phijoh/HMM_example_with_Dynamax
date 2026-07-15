# Setup guide for the GLM-HMM notebook

This guide helps students create a fresh Python environment and install everything needed to run:

- `code/fit_GLMHMM.ipynb`

---

## 1) Install Conda first

Install one of these:
- Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Miniforge: https://github.com/conda-forge/miniforge

Then restart your terminal so `conda` is available.

---

## 2) Open a terminal in the project folder

In VS Code (or another code editor):
- Open your project folder (for example `GLMHMM_EXAMPLE`)
- Open Terminal

---

## 3) Create and activate a Conda environment

```bash
conda create -n glmhmm python=3.11 -y
conda activate glmhmm
```

After activation, your terminal should show `(glmhmm)` at the start.

---

## 4) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5) Register the environment for Jupyter

```bash
python -m ipykernel install --user --name glmhmm --display-name "Python (glmhmm)"
```

---

## 6) Start Jupyter

Open the notebook in VSCode or your code editor of choice.
- `code/fit_GLMHMM.ipynb`

In the notebook, choose kernel:
- **glmhmm (Python 3.11)**

---

## 7) Quick test

Run this in a terminal to verify key imports:

```bash
python -c "import numpy, pandas, matplotlib, seaborn, sklearn, jax, dynamax; print('Setup OK')"
```

---

## Troubleshooting

- If `conda` is not found, restart your terminal and run `conda init` once, then reopen the terminal.
- If activation fails in VS Code, run:
  ```bash
  conda init
  ```
  then restart VS Code.
- If `dynamax` fails to install, update pip first and retry:
  ```bash
  python -m pip install --upgrade pip setuptools wheel
  pip install dynamax
  ```
- If `python` is not found after activation, run `conda activate glmhmm` again.
- If Jupyter does not show the new kernel, restart VS Code and reopen the notebook.
