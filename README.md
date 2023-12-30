# Robotics MVA Homework and LQT,iLQR practice in robotic 2023

This repository contains tutorial notebooks for the 2023 [Robotics](https://www.master-mva.com/cours/robotics/) class at MVA.

## Get started

Clone this repository:

```bash
git clone https://github.com/stephane-caron/robotics-mva-2023.git
```

Install miniconda:

- Linux: https://docs.conda.io/en/latest/miniconda.html
- macOS: https://docs.conda.io/en/latest/miniconda.html
- Windows: https://www.anaconda.com/download/

Don't forget to add the conda snippet to your shell configuration (for instance ``~/.bashrc``). After that, you can run all labs in a dedicated Python environment that will not affect your system's regular Python envirfonment.

### Run a notebook

- Go to your local copy of the repository.
- Open a terminal.
- Create the conda environment:

```bash
conda env create -f robotics-mva.yml
```

From there on, to work on a notebook, you will only need to activate the environment:

```bash
conda activate robotics-mva
```

Then launch the notebook with:

```bash
jupyter-lab
```

The notebook will be accessible from your web browser at [localhost:8888](http://localhost:8888).

Meshcat visualisation can be accessed in full page at `localhost:700N/static/` where N denotes the Nth MeshCat instance created by your notebook kernel.

## Troubleshooting

- Make sure the virtual environment is activated for ``jupyter-lab`` to work.
- Make sure you call ``jupyter-lab`` so that Python packages pathes are configured properly.
    - In particular, ``jupyter-notebook`` may not have paths configured properly, resulting in failed package imports.
