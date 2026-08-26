"""
Career Progression & Promotion Gap Analysis — Streamlit dashboard.

Reads data/palo_alto_networks_scored.csv (written by the last cell of
career_progression_analysis.ipynb). Run the notebook first if that file
doesn't exist yet.

    streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Career Progression & Promotion Gap Analysis",
    layout="wide",
)

DATA_PATH = Path("data") / "palo_alto_networks_scored.csv"

CLUSTER_ORDER = [
    "Fast-track performers",
    "Early-career explorers",
    "Stable long-term contributors",
    "High-risk stagnation profiles",
]
RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLORS = {"Low": "#2E86AB", "Medium": "#F4A261", "High": "#E76F51"}


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["PromotionGapScore"] = pd.Categorical(df["PromotionGapScore"], categories=RISK_ORDER, ordered=True)
    return df


if not DATA_PATH.exists():
    st.error(
        f"Couldn't find `{DATA_PATH}`. Run career_progression_analysis.ipynb "
        "end-to-end first — its last cell writes this file."
    )
    st.stop()

df = load_data(DATA_PATH)

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Filters")

departments = sorted(df["Department"].unique())
dept_filter = st.sidebar.multiselect("Department", departments, default=departments)

clusters_present = [c for c in CLUSTER_ORDER if c in df["CareerCluster"].unique()]
cluster_filter = st.sidebar.multiselect("Career cluster", clusters_present, default=clusters_present)

risk_filter = st.sidebar.multiselect("Promotion gap score", RISK_ORDER, default=RISK_ORDER)

attrition_filter = st.sidebar.radio("Attrition status", ["All", "Stayed only", "Left only"], index=0)

fdf = df[
    df["Department"].isin(dept_filter)
    & df["CareerCluster"].isin(cluster_filter)
    & df["PromotionGapScore"].isin(risk_filter)
]
if attrition_filter == "Stayed only":
    fdf = fdf[fdf["Attrition"] == 0]
elif attrition_filter == "Left only":
    fdf = fdf[fdf["Attrition"] == 1]

# ---------------------------------------------------------------- header
st.title("Career Progression & Promotion Gap Analysis")
st.caption(
    "Palo Alto Networks HR extract — clustering employees by promotion gaps and "
    "role stagnation instead of only predicting who leaves."
)

# ---------------------------------------------------------------- KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Employees (filtered)", f"{len(fdf):,}", help=f"of {len(df):,} total")
k2.metric("High promotion-gap risk", f"{(fdf['PromotionGapScore'] == 'High').mean() * 100:.1f}%" if len(fdf) else "—")
n_opportunity = int(fdf["RetentionOpportunity"].sum()) if len(fdf) else 0
k3.metric("Retention opportunities", f"{n_opportunity:,}")
k4.metric("Attrition rate (filtered)", f"{fdf['Attrition'].mean() * 100:.1f}%" if len(fdf) else "—")

st.divider()

# ---------------------------------------------------------------- row 1: clusters + risk
c1, c2 = st.columns(2)

with c1:
    st.subheader("Headcount by career cluster")
    cluster_counts = (
        fdf["CareerCluster"].value_counts().reindex(clusters_present).fillna(0).reset_index()
    )
    cluster_counts.columns = ["CareerCluster", "Count"]
    fig = px.bar(
        cluster_counts, x="Count", y="CareerCluster", orientation="h", color="CareerCluster",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Employees")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Promotion gap score distribution")
    risk_counts = fdf["PromotionGapScore"].value_counts().reindex(RISK_ORDER).fillna(0).reset_index()
    risk_counts.columns = ["PromotionGapScore", "Count"]
    fig = px.bar(
        risk_counts, x="PromotionGapScore", y="Count", color="PromotionGapScore",
        color_discrete_map=RISK_COLORS, category_orders={"PromotionGapScore": RISK_ORDER},
    )
    fig.update_layout(showlegend=False, xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- row 2: scatter + dept
c3, c4 = st.columns(2)

with c3:
    st.subheader("Promotion gap vs. role stagnation")
    fig = px.scatter(
        fdf, x="PromotionGapRatio", y="RoleStagnationIndex", color="CareerCluster",
        opacity=0.6, color_discrete_sequence=px.colors.qualitative.Safe,
        hover_data=["Department", "JobRole", "PromotionGapScore"],
    )
    st.plotly_chart(fig, use_container_width=True)

with c4:
    st.subheader("Retention opportunities by department")
    opp_by_dept = (
        fdf[fdf["RetentionOpportunity"]]["Department"].value_counts().reset_index()
    )
    opp_by_dept.columns = ["Department", "Count"]
    if len(opp_by_dept):
        fig = px.bar(opp_by_dept, x="Count", y="Department", orientation="h", color_discrete_sequence=["#6A4C93"])
        fig.update_layout(yaxis_title="", xaxis_title="Employees")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No retention opportunities in the current filter selection.")

st.divider()

# ---------------------------------------------------------------- table
st.subheader("Retention opportunity — employee detail")
st.caption("Still at the company, Medium/High promotion-gap score, job satisfaction ≥ 2.")
opp_table = fdf.loc[
    fdf["RetentionOpportunity"],
    ["EmployeeID", "Department", "JobRole", "CareerCluster", "PromotionGapScore",
     "YearsSinceLastPromotion", "YearsInCurrentRole", "JobSatisfaction"],
].sort_values("YearsSinceLastPromotion", ascending=False)
st.dataframe(opp_table, use_container_width=True, hide_index=True)

with st.expander("Full filtered dataset"):
    st.dataframe(fdf, use_container_width=True, hide_index=True)
