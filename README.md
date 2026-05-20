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

## If you want to build more XRD data:

While (.venv) is active:

```bash
pip install mp-api
pip install pymatgen
```
Customise the contents of /preprocessing/build_xrd_dataset
>   Customise variables in script

Create a custom Materials Project API Key from "https://next-gen.materialsproject.org"
>   Home / API / API Key / "Your API Key"

>   In powershell:
```bash
$env:MP_API_KEY="Your API Key"
```

One downside: downloads can be large. There exists a small dataset (400 entries) by default.

```bash
python -m preprocessing.build_xrd_dataset
```