# DAMC ML, Neural Networks, Deep Learning Workshop

A lightweight CPU-only workshop repo for three progressively modern modelling tasks.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python -m easy_ml.train
python -m medium_nn.train
python -m hard_dl.train
```

Each task writes outputs to `outputs/<task_name>/`.


## If you want to build more data:

While (.venv) is active:

```bash
pip install mp-api
pip install pymatgen
```