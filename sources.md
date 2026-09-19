# Sources and Parameter Documentation

This file should become the audit trail for every empirical parameter used by
the Democratic ROI Simulator.

## Important rule

**Do not replace illustrative values with empirical values unless the original
source has been personally verified.**

For every empirical input, record:

- full citation
- stable URL / DOI
- exact sample
- election/context
- treatment type
- outcome definition
- point estimate
- uncertainty interval
- whether the estimate is causal
- which simulator parameter it informs
- important limitations

---

## Parameter map

### m_g — Immediate mobilization effect

Needed:
- randomized or strong quasi-experimental GOTV estimates
- ideally by age or first-time-voter status
- effects separated by contact method where possible

Current status:
- illustrative only

### c_g — Cost per contact / mobilization

Needed:
- field-experiment or campaign-program cost estimates
- must distinguish cost per attempted contact from cost per successful contact
  and cost per additional vote

Current status:
- illustrative only

### rho_g / p_(g,t) — Persistence

Needed:
- studies of turnout persistence or habit formation
- ideally with multiple post-treatment elections
- investigate whether persistence differs by age / first electoral experience

Current status:
- geometric persistence assumption for exploration only

### s_g — Support probability

Needed:
- this may remain a scenario parameter rather than a fixed empirical quantity
- actual support probabilities vary by candidate, election, place, and group

Current status:
- illustrative only

### beta — Discount factor

Normative / analytical parameter.
Should generally be scenario-based rather than presented as an empirical fact.

### lambda — Organizational internalization

Theoretical parameter representing the share of future electoral value that
an organization effectively values or expects to internalize.

Current status:
- theoretical scenario parameter
- no empirical estimate claimed

---

## Literature review notes

Add verified sources here as they are read.

### Source 1
Citation:
Question:
Data:
Method:
Finding:
Parameter relevance:
Limitations:

### Source 2
Citation:
Question:
Data:
Method:
Finding:
Parameter relevance:
Limitations:
