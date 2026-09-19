# Democratic ROI Simulator — Version 2

A transparent research prototype exploring whether current-election electoral
optimization can rank voter-mobilization investments differently from a
benchmark that values long-run democratic participation.

## What's new in Version 2

- Illustrative Mode
- Literature-Calibrated Research Mode
- empirical source audit trail
- source-backed parameter database
- Monte Carlo uncertainty analysis
- organization-internalization scenarios
- λ crossover analysis
- sensitivity surfaces
- explicit limitations and research-integrity notes
- downloadable Monte Carlo draws

## Important interpretation rule

Research Mode does **not** claim to estimate a causal age effect.

Its youth profile and general registered-voter benchmark come from different
studies and contexts. They are used to explore the implications of
literature-based parameter ranges, not to prove that young voters have a
particular ROI relative to older voters.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Core files

- `app.py` — website
- `model.py` — deterministic formal model
- `simulation.py` — Monte Carlo engine
- `research.py` — source/parameter loader
- `data/parameters.csv` — parameter audit table
- `data/sources.csv` — verified source metadata
- `METHODOLOGY.md` — modeling notes
- `sources.md` — source notes

## Research integrity

The simulator is for aggregate theoretical/research analysis. It is not a voter
targeting tool and does not identify or rank individual voters.
