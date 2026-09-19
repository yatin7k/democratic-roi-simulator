# Democratic ROI Simulator

An interactive research prototype exploring whether political organizations
that optimize for current-election electoral returns can rank voter-mobilization
investments differently from a benchmark that values long-run democratic
participation.

## Core research question

Under what combinations of mobilization effects, contact costs, participation
persistence, time horizons, discounting, and organizational internalization
does a short-run electoral objective diverge from a long-run participation
objective?

## Working concepts

### Democratic Investment Externality

A theoretical mechanism in which an organization pays the present cost of
mobilizing a new participant but internalizes only part of the future electoral
benefit generated if that participant continues voting.

### Democratic ROI Paradox

A ranking reversal in which one voter group has the higher current-election
candidate-specific return while another has the higher long-run democratic
participation return.

These are working concepts developed for this project, not established terms
in the political-science literature.

## Current status

**Version 1 is a research prototype.**

The default parameters are illustrative assumptions, not empirical estimates.
No output should be interpreted as a claim about actual young voters,
established voters, campaigns, or parties.

## Model

Current-election return:

R^C_g = (s_g * m_g) / c_g

Long-run democratic return:

R^D_g =
[m_g + sum(beta^t * p_(g,t))] / c_g

Version 1 persistence assumption:

p_(g,t) = m_g * rho_g^t

Organization-specific return:

R^O_(g,j) =
[s_(g,0)m_g
 + lambda_j * sum(beta^t * s_(g,t) * p_(g,t))]
/ c_g

## Files

- `app.py` — Streamlit interface
- `model.py` — formal model and calculations
- `requirements.txt` — Python dependencies
- `sources.md` — research-source and parameter documentation
- `.gitignore` — files excluded from Git

## Run locally

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

## Run in GitHub Codespaces

Open the repository in a Codespace, then:

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Codespaces should offer to open or forward the Streamlit port.

## Deploy with Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Create a new app.
4. Select this GitHub repository.
5. Set the entry-point file to `app.py`.
6. Deploy.

## Research roadmap

- [x] Basic current-election ROI
- [x] Long-run democratic ROI
- [x] Organizational internalization parameter lambda
- [x] Exact lambda crossover threshold
- [x] Interactive parameter controls
- [x] Lambda ROI visualization
- [x] Sensitivity analysis
- [ ] Literature-calibrated parameter ranges
- [ ] Uncertainty intervals
- [ ] Monte Carlo analysis
- [ ] Non-geometric persistence functions
- [ ] Organization-type comparison
- [ ] Full research manuscript
- [ ] Public methods documentation

## Research integrity

This simulator is for aggregate theoretical analysis. It is not designed to
identify, rank, persuade, or target individual voters.

## License

Add the license you prefer before broad public reuse.
