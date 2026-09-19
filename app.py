from __future__ import annotations

import math
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
from research import load_sources, load_parameters, source_for_parameter
from simulation import (
    DistributionSpec,
    run_monte_carlo,
    summarize_mc,
)


st.set_page_config(
    page_title="Democratic ROI Simulator",
    page_icon="🗳️",
    layout="wide",
)


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def make_group(name, cost, effect_pct, support_pct, persistence_pct):
    return VoterGroup(
        name=name,
        cost=float(cost),
        mobilization_effect=float(effect_pct) / 100.0,
        support_probability=float(support_pct) / 100.0,
        persistence=float(persistence_pct) / 100.0,
    )


def dist_from_row(row: dict) -> DistributionSpec:
    def val(key):
        x = row.get(key)
        if pd.isna(x):
            return None
        return float(x)

    return DistributionSpec(
        distribution=str(row["distribution"]),
        low=float(row["low"]),
        high=float(row["high"]),
        mean=val("mean"),
        sd=val("sd"),
        mode=val("mode"),
    )


def research_parameter(parameter_id: str) -> dict:
    df = load_parameters()
    row = df[df["parameter_id"] == parameter_id]
    if row.empty:
        raise KeyError(parameter_id)
    return row.iloc[0].to_dict()


def render_source_expander(parameter_id: str):
    joined = source_for_parameter(parameter_id)
    if joined.empty:
        return

    row = joined.iloc[0]
    with st.expander(f"Source & assumptions — {row['display_name']}"):
        st.markdown(f"**Source:** {row['citation']}")
        st.markdown(f"**Context:** {row['context']}")
        st.markdown(f"**Model use:** {row['model_use']}")
        st.markdown(f"**Limitation:** {row['limitation']}")
        st.markdown(f"**URL:** {row['url']}")


def org_scenario_ui():
    scenario = st.sidebar.selectbox(
        "Organizational scenario",
        [
            "Custom λ",
            "Short-horizon candidate-centered scenario",
            "Intermediate internalization scenario",
            "High-internalization durable-organization scenario",
        ],
        help=(
            "These λ presets are analytical scenarios, not empirical "
            "estimates of real organization types."
        ),
    )

    if scenario == "Custom λ":
        lam = st.sidebar.slider(
            "λ — organizational internalization",
            0.0, 1.0, 0.50, 0.01,
        )
    elif scenario == "Short-horizon candidate-centered scenario":
        lam = 0.15
        st.sidebar.caption("Analytical preset: λ = 0.15")
    elif scenario == "Intermediate internalization scenario":
        lam = 0.50
        st.sidebar.caption("Analytical preset: λ = 0.50")
    else:
        lam = 0.85
        st.sidebar.caption("Analytical preset: λ = 0.85")

    st.sidebar.caption(
        "λ has not been empirically estimated here. Presets are "
        "theoretical sensitivity scenarios only."
    )
    return scenario, lam


def results_sentence(comparison, threshold, lam, y_org, e_org):
    paradox = comparison["democratic_roi_paradox"]

    if paradox:
        first = (
            "Under the selected assumptions, the current-election and "
            "long-run participation objectives rank the two profiles differently."
        )
    else:
        first = (
            "Under the selected assumptions, the current-election and "
            "long-run participation objectives do not produce a ranking reversal."
        )

    org_winner = (
        "Young / First-Time profile"
        if y_org > e_org
        else "General / Established benchmark"
        if e_org > y_org
        else "Neither profile (tie)"
    )

    second = (
        f"At λ={lam:.2f}, the organization-specific model prefers "
        f"**{org_winner}**."
    )

    if threshold is None:
        third = (
            "No organizational λ crossover occurs inside [0,1] under "
            "these point assumptions."
        )
    else:
        third = (
            f"The point-estimate organizational crossover is "
            f"λ* ≈ **{threshold:.3f}**."
        )

    return first + " " + second + " " + third


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("Democratic ROI Simulator")

st.markdown(
    """
A transparent research prototype for asking when **current-election
electoral optimization** and **long-run democratic participation**
produce different rankings of mobilization investments.
"""
)

mode = st.radio(
    "Model mode",
    ["Illustrative Mode", "Literature-Calibrated Research Mode"],
    horizontal=True,
)

if mode == "Illustrative Mode":
    st.warning(
        "Illustrative Mode uses user-selected assumptions. "
        "Nothing in this mode is an empirical claim."
    )
else:
    st.info(
        "Research Mode uses verified literature-based inputs where possible. "
        "It is still exploratory: the two profiles come from different studies "
        "and are NOT a clean causal age comparison."
    )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Global model settings")

beta = st.sidebar.slider(
    "β — discount factor",
    0.0, 1.0, 0.95, 0.01,
)

horizon = st.sidebar.slider(
    "T — future elections",
    0, 10, 4, 1,
)

scenario_name, lam = org_scenario_ui()

st.sidebar.divider()

if mode == "Illustrative Mode":
    st.sidebar.header("Young / First-Time Voters")

    y_cost = st.sidebar.number_input(
        "Young cost/contact ($)",
        min_value=0.01,
        value=5.00,
        step=0.25,
    )
    y_effect = st.sidebar.slider(
        "Young immediate turnout effect (%)",
        0.0, 20.0, 3.0, 0.1,
    )
    y_support = st.sidebar.slider(
        "Young support probability (%)",
        0.0, 100.0, 55.0, 1.0,
    )
    y_persistence = st.sidebar.slider(
        "Young persistence ρ (%)",
        0.0, 100.0, 60.0, 1.0,
    )

    st.sidebar.divider()
    st.sidebar.header("Established Voters")

    e_cost = st.sidebar.number_input(
        "Established cost/contact ($)",
        min_value=0.01,
        value=5.00,
        step=0.25,
    )
    e_effect = st.sidebar.slider(
        "Established immediate turnout effect (%)",
        0.0, 20.0, 5.0, 0.1,
    )
    e_support = st.sidebar.slider(
        "Established support probability (%)",
        0.0, 100.0, 55.0, 1.0,
    )
    e_persistence = st.sidebar.slider(
        "Established persistence ρ (%)",
        0.0, 100.0, 15.0, 1.0,
    )

else:
    st.sidebar.header("Research-mode scenario inputs")

    y_support = st.sidebar.slider(
        "Youth-profile support probability (%)",
        0.0, 100.0, 55.0, 1.0,
        help="Scenario variable; not estimated by the cited GOTV studies.",
    )
    e_support = st.sidebar.slider(
        "General-benchmark support probability (%)",
        0.0, 100.0, 55.0, 1.0,
        help="Scenario variable; not estimated by the cited GOTV studies.",
    )

    # Point values for deterministic display use midpoints / means.
    y_eff_row = research_parameter("youth_canvass_effect")
    y_cpv_row = research_parameter("youth_canvass_cost_per_additional_vote")
    e_eff_row = research_parameter("general_canvass_effect")
    e_cost_row = research_parameter("general_canvass_contact_cost")
    pers_row = research_parameter("habit_multiplier")

    y_effect = 100 * (
        float(y_eff_row["mean"])
        if not pd.isna(y_eff_row["mean"])
        else (float(y_eff_row["low"]) + float(y_eff_row["high"])) / 2
    )
    y_cpv_mid = (
        float(y_cpv_row["low"]) + float(y_cpv_row["high"])
    ) / 2
    y_cost = (y_effect / 100.0) * y_cpv_mid

    e_effect = 100 * float(e_eff_row["mean"])
    e_cost = float(e_cost_row["mean"])

    persistence_mid = (
        float(pers_row["low"]) + float(pers_row["high"])
    ) / 2
    y_persistence = 100 * persistence_mid
    e_persistence = 100 * persistence_mid

    st.sidebar.caption(
        "Point display uses literature midpoints/means. Monte Carlo "
        "uses the stored distributions instead."
    )


young_name = (
    "Young / First-Time profile"
    if mode == "Literature-Calibrated Research Mode"
    else "Young / First-Time Voters"
)
established_name = (
    "General / Established benchmark"
    if mode == "Literature-Calibrated Research Mode"
    else "Established Voters"
)

young = make_group(
    young_name,
    y_cost,
    y_effect,
    y_support,
    y_persistence,
)
established = make_group(
    established_name,
    e_cost,
    e_effect,
    e_support,
    e_persistence,
)


# ---------------------------------------------------------
# Core outputs
# ---------------------------------------------------------

comparison = compare_groups(young, established, beta, horizon)
threshold = exact_lambda_threshold(young, established, beta, horizon)

y_current = current_election_roi(young)
e_current = current_election_roi(established)
y_demo = democratic_roi(young, beta, horizon)
e_demo = democratic_roi(established, beta, horizon)
y_org = organizational_roi(young, lam, beta, horizon)
e_org = organizational_roi(established, lam, beta, horizon)


tabs = st.tabs(
    [
        "Results",
        "Simulator",
        "λ Threshold",
        "Monte Carlo",
        "Sensitivity",
        "Sources",
        "Methodology",
        "About",
    ]
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

with tabs[0]:
    st.subheader("Results Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Current-election preference",
        comparison["current_winner"],
    )
    c2.metric(
        "Long-run participation preference",
        comparison["democratic_winner"],
    )
    c3.metric(
        "Democratic ROI Paradox",
        "Detected"
        if comparison["democratic_roi_paradox"]
        else "Not detected",
    )

    st.markdown(
        results_sentence(
            comparison, threshold, lam, y_org, e_org
        )
    )

    if mode == "Literature-Calibrated Research Mode":
        st.warning(
            "Interpret this as a literature-calibrated comparison of two "
            "study profiles, not evidence that age itself causes the ROI gap. "
            "The cited studies differ in campaign, place, year, and design."
        )

    summary_df = pd.DataFrame(
        {
            "Profile": [young.name, established.name],
            "Current-election ROI": [y_current, e_current],
            "Long-run democratic ROI": [y_demo, e_demo],
            f"Organizational ROI (λ={lam:.2f})": [y_org, e_org],
        }
    )
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# Simulator
# ---------------------------------------------------------

with tabs[1]:
    st.subheader("Point-Estimate Simulator")

    point_df = pd.DataFrame(
        {
            young.name: [y_current, y_demo, y_org],
            established.name: [e_current, e_demo, e_org],
        },
        index=[
            "Current-election ROI",
            "Long-run democratic ROI",
            f"Organizational ROI at λ={lam:.2f}",
        ],
    )
    st.bar_chart(point_df)

    st.caption(
        "Point estimates summarize one selected scenario. Research Mode "
        "should be interpreted together with Monte Carlo uncertainty."
    )

    if mode == "Literature-Calibrated Research Mode":
        st.subheader("Research-mode inputs used for point display")
        inputs = pd.DataFrame(
            {
                "Quantity": [
                    "Youth effect",
                    "Youth derived cost/contact",
                    "General effect",
                    "General contact cost",
                    "Persistence multiplier (both)",
                ],
                "Value": [
                    f"{y_effect:.2f} percentage points",
                    f"${y_cost:.2f}",
                    f"{e_effect:.2f} percentage points",
                    f"${e_cost:.2f}",
                    f"{y_persistence:.2f}%",
                ],
            }
        )
        st.dataframe(inputs, hide_index=True, use_container_width=True)

        st.subheader("Parameter source audit")
        render_source_expander("youth_canvass_effect")
        render_source_expander("youth_canvass_cost_per_additional_vote")
        render_source_expander("general_canvass_effect")
        render_source_expander("general_canvass_contact_cost")
        render_source_expander("habit_multiplier")


# ---------------------------------------------------------
# Lambda threshold
# ---------------------------------------------------------

with tabs[2]:
    st.subheader("Organizational ROI as λ Changes")

    lambdas = np.linspace(0, 1, 201)
    curve = pd.DataFrame(
        {
            "λ": lambdas,
            young.name: roi_curve(
                young, lambdas.tolist(), beta, horizon
            ),
            established.name: roi_curve(
                established, lambdas.tolist(), beta, horizon
            ),
        }
    ).set_index("λ")

    st.line_chart(curve)

    if threshold is None:
        st.info(
            "No point-estimate λ crossover occurs inside [0,1]."
        )
    else:
        st.success(
            f"Point-estimate crossover: λ* ≈ {threshold:.4f}"
        )

    st.caption(
        "λ is a theoretical internalization parameter. The organization "
        "scenario presets are analytical aids, not estimated real-world λ values."
    )


# ---------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------

with tabs[3]:
    st.subheader("Monte Carlo Uncertainty Analysis")

    if mode != "Literature-Calibrated Research Mode":
        st.info(
            "Switch to Literature-Calibrated Research Mode to run the "
            "source-based Monte Carlo analysis."
        )
    else:
        n_sims = st.select_slider(
            "Number of simulations",
            options=[1000, 5000, 10000, 25000],
            value=10000,
        )
        seed = st.number_input(
            "Random seed",
            min_value=0,
            value=42,
            step=1,
        )

        y_eff = dist_from_row(
            research_parameter("youth_canvass_effect")
        )
        y_cpv = dist_from_row(
            research_parameter(
                "youth_canvass_cost_per_additional_vote"
            )
        )
        e_eff = dist_from_row(
            research_parameter("general_canvass_effect")
        )
        e_cost_spec = dist_from_row(
            research_parameter("general_canvass_contact_cost")
        )
        persistence_spec = dist_from_row(
            research_parameter("habit_multiplier")
        )

        with st.spinner("Running Monte Carlo simulations..."):
            mc = run_monte_carlo(
                young_effect_spec=y_eff,
                young_cost_mode="cpv_derived",
                young_cost_spec=y_cpv,
                general_effect_spec=e_eff,
                general_cost_spec=e_cost_spec,
                persistence_spec=persistence_spec,
                young_support=y_support / 100.0,
                general_support=e_support / 100.0,
                beta=beta,
                horizon=horizon,
                lambda_internalization=lam,
                n=int(n_sims),
                seed=int(seed),
            )
            sm = summarize_mc(mc)

        a, b, c = st.columns(3)
        a.metric(
            "Ranking-reversal rate",
            f"{100*sm['paradox_rate']:.1f}%",
        )
        b.metric(
            f"P(Youth profile preferred at λ={lam:.2f})",
            f"{100*sm['young_org_preferred_rate']:.1f}%",
        )
        c.metric(
            "Runs with λ crossover in [0,1]",
            f"{100*sm['lambda_crossing_rate']:.1f}%",
        )

        if not math.isnan(sm["lambda_median"]):
            st.markdown(
                f"Conditional on a crossover occurring, median "
                f"λ* = **{sm['lambda_median']:.3f}**, with a "
                f"10th–90th percentile interval of "
                f"**[{sm['lambda_p10']:.3f}, {sm['lambda_p90']:.3f}]**."
            )
        else:
            st.markdown(
                "No valid λ* values occurred in these simulations."
            )

        st.warning(
            "These percentages describe uncertainty inside this model and "
            "its chosen literature-calibration rules. They are not direct "
            "probabilities that real campaigns behave this way."
        )

        st.subheader("Distribution of λ*")

        valid_lambda = mc["lambda_star"].dropna()
        if len(valid_lambda):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(valid_lambda, bins=30)
            ax.set_xlabel("λ*")
            ax.set_ylabel("Simulation count")
            ax.set_title("Monte Carlo Distribution of Organizational Crossover λ*")
            st.pyplot(fig, use_container_width=True)
        else:
            st.caption("No λ* values to plot.")

        st.subheader("Probability Youth Profile Has Higher Organizational ROI Across λ")

        lambda_curve = np.linspace(0.0, 1.0, 101)
        probability_curve = []

        y_i = mc["young_org_intercept"].to_numpy()
        y_s = mc["young_org_slope"].to_numpy()
        e_i = mc["general_org_intercept"].to_numpy()
        e_s = mc["general_org_slope"].to_numpy()

        for lval in lambda_curve:
            prob = np.mean(
                (y_i + lval * y_s) >
                (e_i + lval * e_s)
            )
            probability_curve.append(prob)

        probability_df = pd.DataFrame(
            {
                "λ": lambda_curve,
                "P(Youth profile has higher organizational ROI)": probability_curve,
            }
        ).set_index("λ")

        st.line_chart(probability_df)

        st.caption(
            "This curve is conditional on the Version 2 model, evidence bands, "
            "support-probability scenario, β, and horizon. It is not a direct "
            "estimate of real-world campaign behavior."
        )

        st.subheader("ROI-gap uncertainty")

        gap_df = pd.DataFrame(
            {
                "Current-election gap (Youth - General)": (
                    mc["young_current_roi"] -
                    mc["general_current_roi"]
                ),
                "Long-run gap (Youth - General)": (
                    mc["young_democratic_roi"] -
                    mc["general_democratic_roi"]
                ),
                "Organizational gap (Youth - General)": (
                    mc["young_org_roi"] -
                    mc["general_org_roi"]
                ),
            }
        )

        st.dataframe(
            gap_df.describe(percentiles=[0.1, 0.5, 0.9]).T,
            use_container_width=True,
        )

        csv = mc.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Monte Carlo draws (CSV)",
            csv,
            file_name="democratic_roi_monte_carlo.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------

with tabs[4]:
    st.subheader("Sensitivity Analysis")

    persistence_grid = np.linspace(0, 1, 51)
    lambda_grid = np.linspace(0, 1, 51)
    matrix = np.zeros((51, 51))

    for i, persistence in enumerate(persistence_grid):
        y_temp = VoterGroup(
            young.name,
            young.cost,
            young.mobilization_effect,
            young.support_probability,
            float(persistence),
        )
        for j, l in enumerate(lambda_grid):
            y_val = organizational_roi(
                y_temp, float(l), beta, horizon
            )
            e_val = organizational_roi(
                established, float(l), beta, horizon
            )
            matrix[i, j] = 1 if y_val > e_val else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=[0, 1, 0, 1],
    )
    ax.set_xlabel("λ — organizational internalization")
    ax.set_ylabel("Youth-profile persistence ρ")
    ax.set_title(
        "Region Where Youth Profile Has Higher Organizational ROI"
    )
    fig.colorbar(
        im,
        ax=ax,
        ticks=[0, 1],
        label="0 = General benchmark preferred; 1 = Youth preferred",
    )
    st.pyplot(fig, use_container_width=True)

    st.markdown(
        """
This plot is deliberately a **sensitivity surface**. It does not claim
that high values of youth persistence are empirically established.
Its purpose is to show which assumptions drive the model's ranking.
"""
    )

    horizons = list(range(0, 11))
    hdf = pd.DataFrame(
        {
            "Future elections": horizons,
            young.name: [
                democratic_roi(young, beta, h) for h in horizons
            ],
            established.name: [
                democratic_roi(established, beta, h)
                for h in horizons
            ],
        }
    ).set_index("Future elections")

    st.subheader("Sensitivity to analytical time horizon")
    st.line_chart(hdf)


# ---------------------------------------------------------
# Sources
# ---------------------------------------------------------

with tabs[5]:
    st.subheader("Empirical Source Audit Trail")

    st.markdown(
        """
Research Mode is intentionally conservative. Each empirical input
below is tied to a source, while **β, λ, and candidate support
probabilities remain analytical/scenario parameters**.

The two mobilization profiles are not from a single randomized
young-vs.-old comparison, so the app does not attribute profile
differences causally to age.
"""
    )

    params = load_parameters()
    sources = load_sources()

    for _, p in params.iterrows():
        st.markdown(f"### {p['display_name']}")
        st.markdown(f"**Model use:** {p['model_use']}")
        st.markdown(
            f"**Distribution:** {p['distribution']} | "
            f"range [{p['low']}, {p['high']}]"
        )
        if not pd.isna(p["mean"]):
            st.markdown(f"**Mean/point:** {p['mean']}")
        if not pd.isna(p["sd"]):
            st.markdown(f"**SD/SE used:** {p['sd']}")

        source = sources[sources["source_id"] == p["source_id"]]
        if not source.empty:
            s = source.iloc[0]
            with st.expander("Study details"):
                st.markdown(f"**Citation:** {s['citation']}")
                st.markdown(f"**Context:** {s['context']}")
                st.markdown(f"**Key result:** {s['key_result']}")
                st.markdown(f"**Limitation:** {s['limitation']}")
                st.markdown(f"**URL:** {s['url']}")

    st.subheader("Full source table")
    st.dataframe(sources, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# Methodology
# ---------------------------------------------------------

with tabs[6]:
    st.subheader("Formal Model")

    st.markdown("### Current-election candidate-specific return")
    st.latex(r"R_g^C = \frac{s_g m_g}{c_g}")

    st.markdown(
        r"""
where \(s_g\) is support probability, \(m_g\) is the immediate
mobilization effect, and \(c_g\) is mobilization cost.
"""
    )

    st.markdown("### Geometric downstream-effect assumption")
    st.latex(r"p_{g,t} = m_g \rho_g^t")

    st.markdown(
        r"""
where \(\rho_g\) is the persistence multiplier for voter group \(g\).
"""
    )

    st.markdown("### Long-run democratic-participation return")
    st.latex(
        r"R_g^D = \frac{m_g + \sum_{t=1}^{T}\beta^t p_{g,t}}{c_g}"
    )

    st.markdown("### Organization-specific return")
    st.latex(
        r"R_{g,j}^O = "
        r"\frac{s_{g,0}m_g + "
        r"\lambda_j\sum_{t=1}^{T}\beta^t s_{g,t}p_{g,t}}{c_g}"
    )

    st.markdown("### Internalization parameter")
    st.latex(r"\lambda_j \in [0,1]")

    st.markdown(
        r"""
\(\lambda_j\) represents the share of future electoral value that
organization \(j\) effectively values or expects to internalize.
"""
    )

    st.markdown("### Ranking-reversal definition")
    st.latex(
        r"R_E^C > R_Y^C"
        r"\qquad\text{while}\qquad"
        r"R_Y^D > R_E^D"
    )

    st.markdown(
        """
The project calls a **Democratic ROI Paradox** a case in which two
profiles receive opposite rankings under the current-election and
long-run participation objectives.
"""
    )

    st.subheader("How Research Mode is calibrated")

    st.markdown(
        r"""
**Youth mobilization effect.** The youth-canvassing profile uses a
7–10 percentage-point range reported in a Pew/CIRCLE synthesis of
randomized youth GOTV field experiments.

**Youth contact cost.** The same report gives $11–$14 per additional
vote for youth door-to-door canvassing. Because the formal model needs
cost per contact/attempt, Research Mode derives a corresponding
per-contact cost in each Monte Carlo draw using the relationship below.
"""
    )

    st.latex(
        r"c_Y = m_Y \times "
        r"\left(\text{cost per additional vote}\right)"
    )

    st.markdown(
        r"""
This is a derived approximation and is labeled as such.

**General canvassing benchmark.** Gerber & Green's New Haven field
experiment supplies a broad registered-voter canvassing benchmark.
The research-mode distribution uses the reported personal-contact
effect estimate and uncertainty, while contact cost uses the study's
reported approximate $1.50 personal-contact cost.

**Persistence.** Coppock & Green conclude from experiments and
regression-discontinuity evidence that voting is habit-forming and
report an average effect of roughly 10 percentage points, with
important contextual heterogeneity. Version 2 therefore uses a
conservative 0.05–0.10 analytical evidence band for the persistence
multiplier in Monte Carlo rather than claiming a precise age-specific
persistence estimate.

Crucially, the same persistence evidence band is used for both
profiles by default. Version 2 therefore **does not assume** that
young voters have stronger persistence merely because the theory
would benefit from that assumption.
"""
    )

    st.subheader("Monte Carlo logic")
    st.markdown(
        """
Each simulation draws uncertain inputs from the stored distributions,
recomputes all three ROI measures, tests whether the rankings reverse,
and solves for λ* when an organizational crossover exists inside
[0,1].

The Monte Carlo answers:

> How often does the model produce a ranking reversal under the
> specified evidence bands and analytical assumptions?

It does **not** answer:

> What is the real-world probability that campaigns underinvest in
> young voters?

That stronger claim would require a much richer empirical design.
"""
    )

    st.subheader("Important limitations")
    st.markdown(
        """
- The two literature profiles differ across studies, years, places,
  and organizational settings; they are not an age-only comparison.
- λ is theoretical and currently unobserved.
- β is normative/analytical rather than empirical.
- Candidate support probabilities are scenario inputs.
- Geometric persistence is a simplifying structure.
- Campaigns optimize over more than turnout: persuasion,
  fundraising, volunteer recruitment, geography, coalition-building,
  information, and legal constraints are omitted.
- Historical cost figures are not inflation-adjusted in Version 2;
  the model uses them as study-context cost inputs rather than claims
  about 2026 campaign prices.
"""
    )


# ---------------------------------------------------------
# About
# ---------------------------------------------------------

with tabs[7]:
    st.subheader("About the Project")

    st.markdown(
        """
The Democratic ROI Simulator is a public companion to an independent
research project on the political economy of voter mobilization.

**Democratic Investment Externality** is the project's working term
for a possible intertemporal incentive problem: an organization may
bear the present cost of creating a participant while internalizing
only part of the future electoral value generated by that
participation.

**Democratic ROI Paradox** is the project's working term for a
ranking reversal between current-election electoral return and a
long-run participation benchmark.

These labels are original working concepts for this project and
should not be represented as established political-science terms.

The simulator is designed for **aggregate theoretical and research
analysis**. It is not built to identify, rank, persuade, or target
individual voters.
"""
    )
