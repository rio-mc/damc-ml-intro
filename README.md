# DAMC ML, Neural Networks, Deep Learning Workshop

A lightweight CPU-only workshop repo for three progressively harder modelling tasks.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python easy_ml/train.py --config easy_ml/config.yaml
python medium_nn/train.py --config medium_nn/config.yaml
python hard_dl/train.py --config hard_dl/config.yaml
```

Each task writes outputs to `outputs/<task_name>/`.
