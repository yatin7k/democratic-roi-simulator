from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from model import (
    VoterGroup,
    current_election_roi,
    democratic_roi,
    organizational_roi,
    compare_groups,
    exact_lambda_threshold,
    roi_curve,
)


st.set_page_config(
    page_title="Democratic ROI Simulator",
    page_icon="🗳️",
    layout="wide",
)


# ---------------------------
# Helpers
# ---------------------------

def pct(x: float) -> str:
    return f"{100*x:.2f}%"


def num(x: float) -> str:
    return f"{x:.6f}"


def make_group(
    name: str,
    cost: float,
    effect_pct: float,
    support_pct: float,
    persistence_pct: float,
) -> VoterGroup:
    return VoterGroup(
        name=name,
        cost=cost,
        mobilization_effect=effect_pct / 100.0,
        support_probability=support_pct / 100.0,
        persistence=persistence_pct / 100.0,
    )


def sensitivity_matrix(
    young_template: VoterGroup,
    established: VoterGroup,
    beta: float,
    horizon: int,
    lambda_values: np.ndarray,
    persistence_values: np.ndarray,
) -> np.ndarray:
    """
    1  => young ROI > established ROI
    0  => established ROI >= young ROI
    """
    out = np.zeros((len(persistence_values), len(lambda_values)))

    for i, persistence in enumerate(persistence_values):
        young = VoterGroup(
            name=young_template.name,
            cost=young_template.cost,
            mobilization_effect=young_template.mobilization_effect,
            support_probability=young_template.support_probability,
            persistence=float(persistence),
        )

        for j, lam in enumerate(lambda_values):
            y = organizational_roi(
                young, float(lam), beta, horizon
            )
            e = organizational_roi(
                established, float(lam), beta, horizon
            )
            out[i, j] = 1 if y > e else 0

    return out


# ---------------------------
# Header
# ---------------------------

st.title("Democratic ROI Simulator")

st.markdown(
    """
**Research question:** What happens when the strategy that maximizes
candidate-specific electoral return today differs from the strategy that
maximizes democratic participation across future elections?

This research prototype compares stylized voter groups under competing
objective functions and shows how organizational time horizons,
mobilization persistence, and future-benefit internalization can change
which investment appears optimal.
"""
)

st.warning(
    """
**Research prototype — illustrative parameters.**
The default numbers below are not empirical estimates and should not be
interpreted as claims about actual young or established voters. The model
is designed to explore assumptions and identify conditions under which
rankings can reverse.
"""
)


# ---------------------------
# Sidebar inputs
# ---------------------------

st.sidebar.header("Global Parameters")

beta = st.sidebar.slider(
    "β — Future participation discount factor",
    min_value=0.0,
    max_value=1.0,
    value=0.95,
    step=0.01,
    help="How strongly future participation benefits are weighted."
)

horizon = st.sidebar.slider(
    "T — Future elections",
    min_value=0,
    max_value=10,
    value=4,
    step=1,
    help="Number of future elections included in the model."
)

lam = st.sidebar.slider(
    "λ — Organizational internalization",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01,
    help=(
        "Share of future electoral value the organization effectively "
        "internalizes."
    )
)

st.sidebar.divider()
st.sidebar.header("Young / First-Time Voters")

y_cost = st.sidebar.number_input(
    "Young: cost per contact ($)",
    min_value=0.01,
    value=5.00,
    step=0.25,
)

y_effect = st.sidebar.slider(
    "Young: immediate turnout effect (%)",
    min_value=0.0,
    max_value=20.0,
    value=3.0,
    step=0.1,
)

y_support = st.sidebar.slider(
    "Young: support probability (%)",
    min_value=0.0,
    max_value=100.0,
    value=55.0,
    step=1.0,
)

y_persistence = st.sidebar.slider(
    "Young: persistence ρ (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=1.0,
)

st.sidebar.divider()
st.sidebar.header("Established Voters")

e_cost = st.sidebar.number_input(
    "Established: cost per contact ($)",
    min_value=0.01,
    value=5.00,
    step=0.25,
)

e_effect = st.sidebar.slider(
    "Established: immediate turnout effect (%)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.1,
)

e_support = st.sidebar.slider(
    "Established: support probability (%)",
    min_value=0.0,
    max_value=100.0,
    value=55.0,
    step=1.0,
)

e_persistence = st.sidebar.slider(
    "Established: persistence ρ (%)",
    min_value=0.0,
    max_value=100.0,
    value=15.0,
    step=1.0,
)


young = make_group(
    "Young / First-Time Voters",
    y_cost,
    y_effect,
    y_support,
    y_persistence,
)

established = make_group(
    "Established Voters",
    e_cost,
    e_effect,
    e_support,
    e_persistence,
)


# ---------------------------
# Core calculations
# ---------------------------

comparison = compare_groups(
    young,
    established,
    beta=beta,
    horizon=horizon,
)

y_current = current_election_roi(young)
e_current = current_election_roi(established)

y_demo = democratic_roi(young, beta, horizon)
e_demo = democratic_roi(established, beta, horizon)

y_org = organizational_roi(
    young,
    lambda_internalization=lam,
    beta=beta,
    horizon=horizon,
)

e_org = organizational_roi(
    established,
    lambda_internalization=lam,
    beta=beta,
    horizon=horizon,
)

threshold = exact_lambda_threshold(
    young,
    established,
    beta=beta,
    horizon=horizon,
)


# ---------------------------
# Tabs
# ---------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Simulator",
        "λ Threshold",
        "Sensitivity",
        "Methodology",
        "About",
    ]
)


# ---------------------------
# Tab 1: Simulator
# ---------------------------

with tab1:
    st.subheader("Current Scenario")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current-Election Preferred Group",
            comparison["current_winner"],
        )

    with col2:
        st.metric(
            "Long-Run Participation Preferred Group",
            comparison["democratic_winner"],
        )

    with col3:
        paradox_text = (
            "Detected"
            if comparison["democratic_roi_paradox"]
            else "Not detected"
        )
        st.metric(
            "Democratic ROI Paradox",
            paradox_text,
        )

    st.divider()

    results_df = pd.DataFrame(
        {
            "Objective": [
                "Current-election ROI",
                "Current-election ROI",
                "Long-run democratic ROI",
                "Long-run democratic ROI",
                f"Organizational ROI at λ={lam:.2f}",
                f"Organizational ROI at λ={lam:.2f}",
            ],
            "Voter Group": [
                young.name,
                established.name,
                young.name,
                established.name,
                young.name,
                established.name,
            ],
            "ROI": [
                y_current,
                e_current,
                y_demo,
                e_demo,
                y_org,
                e_org,
            ],
        }
    )

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("ROI Comparison")

    chart_df = pd.DataFrame(
        {
            young.name: [y_current, y_demo, y_org],
            established.name: [e_current, e_demo, e_org],
        },
        index=[
            "Current election",
            "Long-run democratic",
            f"Organization (λ={lam:.2f})",
        ],
    )

    st.bar_chart(chart_df)

    st.caption(
        "ROI values are model units per dollar of mobilization cost. "
        "They are only as meaningful as the assumptions supplied."
    )


# ---------------------------
# Tab 2: Lambda Threshold
# ---------------------------

with tab2:
    st.subheader("Organizational ROI as λ Changes")

    lambdas = np.linspace(0.0, 1.0, 201)

    curve_df = pd.DataFrame(
        {
            "λ": lambdas,
            young.name: roi_curve(
                young,
                lambdas.tolist(),
                beta=beta,
                horizon=horizon,
            ),
            established.name: roi_curve(
                established,
                lambdas.tolist(),
                beta=beta,
                horizon=horizon,
            ),
        }
    ).set_index("λ")

    st.line_chart(curve_df)

    if threshold is not None:
        st.success(
            f"Under the current assumptions, the organizational "
            f"preference flips at approximately λ* = {threshold:.4f}."
        )

        before = max(0.0, threshold - 0.01)
        after = min(1.0, threshold + 0.01)

        threshold_df = pd.DataFrame(
            {
                "λ": [before, threshold, after],
                young.name: [
                    organizational_roi(
                        young, before, beta, horizon
                    ),
                    organizational_roi(
                        young, threshold, beta, horizon
                    ),
                    organizational_roi(
                        young, after, beta, horizon
                    ),
                ],
                established.name: [
                    organizational_roi(
                        established, before, beta, horizon
                    ),
                    organizational_roi(
                        established, threshold, beta, horizon
                    ),
                    organizational_roi(
                        established, after, beta, horizon
                    ),
                ],
            }
        )

        st.dataframe(
            threshold_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No λ crossover occurs within [0, 1] under the current "
            "assumptions."
        )

    st.markdown(
        """
**Interpretation:** λ represents the extent to which an organization
effectively values or expects to capture future electoral benefits.
A low-λ organization is modeled as having a short effective electoral
horizon; a high-λ organization places more weight on future returns.
"""
    )


# ---------------------------
# Tab 3: Sensitivity
# ---------------------------

with tab3:
    st.subheader("Sensitivity to Young-Voter Persistence and λ")

    st.markdown(
        """
The heatmap below asks a narrow question: **for which combinations of
young-voter persistence and organizational internalization does the
organizational model prefer young/first-time voters over established
voters?**

All other parameters remain fixed at the values selected in the sidebar.
"""
    )

    lambda_grid = np.linspace(0.0, 1.0, 51)
    persistence_grid = np.linspace(0.0, 1.0, 51)

    matrix = sensitivity_matrix(
        young,
        established,
        beta,
        horizon,
        lambda_grid,
        persistence_grid,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=[0, 1, 0, 1],
    )
    ax.set_xlabel("λ — organizational internalization")
    ax.set_ylabel("Young-voter persistence ρ")
    ax.set_title("Region Where Young / First-Time Voters Have Higher Organizational ROI")
    fig.colorbar(
        image,
        ax=ax,
        ticks=[0, 1],
        label="0 = Established preferred, 1 = Young preferred",
    )
    st.pyplot(fig, use_container_width=True)

    st.subheader("Sensitivity to Time Horizon")

    horizons = list(range(0, 11))

    horizon_df = pd.DataFrame(
        {
            "Future elections": horizons,
            young.name: [
                democratic_roi(young, beta, h)
                for h in horizons
            ],
            established.name: [
                democratic_roi(established, beta, h)
                for h in horizons
            ],
        }
    ).set_index("Future elections")

    st.line_chart(horizon_df)

    st.caption(
        "Sensitivity analysis is exploratory until parameter ranges "
        "are calibrated from credible empirical research."
    )


# ---------------------------
# Tab 4: Methodology
# ---------------------------

with tab4:
    st.subheader("Model Definitions")

    st.markdown(
        r"""
### 1. Current-election candidate-specific return

\[
R_g^C = \frac{s_g m_g}{c_g}
\]

where:

- \(s_g\) = support probability for voter group \(g\)
- \(m_g\) = immediate causal turnout effect of mobilization
- \(c_g\) = mobilization/contact cost

---

### 2. Downstream participation effect

Version 1 uses a geometric persistence assumption:

\[
p_{g,t} = m_g \rho_g^t
\]

where \(\rho_g\) is the persistence parameter.

This is a **modeling assumption**, not an empirical law.

---

### 3. Long-run democratic-participation return

\[
R_g^D
=
\frac{
m_g + \sum_{t=1}^{T}\beta^t p_{g,t}
}{
c_g
}
\]

where:

- \(\beta\) = future-benefit discount factor
- \(T\) = number of future elections included
- \(p_{g,t}\) = downstream participation effect in election \(t\)

---

### 4. Organization-specific return

\[
R_{g,j}^O
=
\frac{
s_{g,0}m_g
+
\lambda_j
\sum_{t=1}^{T}
\beta^t s_{g,t}p_{g,t}
}{
c_g
}
\]

where:

- \(\lambda_j \in [0,1]\) = organization \(j\)'s effective
  internalization of future electoral benefits.

Version 1 assumes future support probability remains constant unless
the model is extended.

---

### 5. Democratic ROI Paradox

For two voter groups \(g\) and \(h\), the paradox is detected when the
current-election and long-run democratic objectives rank the groups
differently.

For example:

\[
R_E^C > R_Y^C
\quad\text{while}\quad
R_Y^D > R_E^D.
\]

This is a **formal definition within this project**, not an established
term in the political-science literature.
"""
    )

    st.subheader("Key Assumptions and Limitations")

    st.markdown(
        """
- The model abstracts from persuasion, fundraising, volunteer
  recruitment, geography, legal constraints, coalition effects,
  information acquisition, and other campaign objectives.
- Persistence is modeled geometrically in Version 1.
- Support probabilities are stylized.
- Contact cost is represented as a single per-contact number.
- The democratic benchmark maximizes participation rather than
  candidate-specific electoral value.
- λ is a theoretical internalization parameter; it is not directly
  observed in the current version.
- Results should be interpreted as **comparative-model outputs**, not
  causal claims about real campaigns.
"""
    )


# ---------------------------
# Tab 5: About
# ---------------------------

with tab5:
    st.subheader("About the Project")

    st.markdown(
        """
The **Democratic ROI Simulator** is a research prototype for studying
an intertemporal political-economy question:

> Can a political organization behave rationally according to its own
> short-run electoral incentives while allocating mobilization resources
> differently from a benchmark that values long-run democratic
> participation?

The project's working concepts are:

**Democratic Investment Externality**  
The possibility that the organization paying the present cost of
mobilizing a new participant internalizes only a fraction of the
electoral value generated by that participant in later elections.

**Democratic ROI Paradox**  
A ranking reversal in which the voter group with the greater
current-election return is not the voter group with the greater
long-run participation return.

### Planned development

1. Replace illustrative parameter values with literature-based ranges.
2. Add uncertainty intervals and Monte Carlo simulation.
3. Compare organizational forms with different effective λ values.
4. Allow non-geometric downstream participation effects.
5. Add reproducible source documentation for every empirical parameter.
6. Link the simulator to a full research manuscript.

### Research integrity

The simulator is designed for **aggregate theoretical analysis**.
It is not intended to identify, rank, persuade, or target individual
voters.
"""
    )
