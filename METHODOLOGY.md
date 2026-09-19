# Methodology

## Objective functions

Current-election candidate-specific ROI:

R_C(g) = s_g * m_g / c_g

Long-run democratic participation ROI:

R_D(g) = [m_g + sum_t(beta^t * p_g,t)] / c_g

Version 2 downstream assumption:

p_g,t = m_g * rho_g^t

Organization-specific ROI:

R_O(g,j) =
[s_g,0*m_g + lambda_j*sum_t(beta^t*s_g,t*p_g,t)] / c_g

## Monte Carlo

Research Mode samples uncertain literature-calibrated inputs and recomputes:
- current-election rankings
- long-run participation rankings
- organization-specific rankings
- ranking reversals
- lambda crossover thresholds

The resulting frequencies are conditional on the model and the chosen
calibration. They are not direct real-world probabilities.

## Calibration caution

The youth profile and general registered-voter benchmark are not taken from a
single age-randomized comparison. Version 2 therefore treats them as study
profiles, not causal age effects.

Persistence is especially uncertain. The simulator uses a conservative
evidence band for a geometric persistence multiplier but explicitly does not
claim that youth persistence exceeds established-voter persistence.
