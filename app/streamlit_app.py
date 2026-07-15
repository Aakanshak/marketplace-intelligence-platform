"""Interactive Marketplace Growth & Pricing Intelligence dashboard."""
from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gmv_analysis import monthly_gmv_decomposition, seller_health, price_elasticity
from src.search_analysis import rank_ctr, search_funnel, cohort_retention
from src.ab_test_cuped import simulate_experiment, required_sample_size

st.set_page_config(page_title="Marketplace Intelligence", page_icon="📈", layout="wide")


@st.cache_data(show_spinner="Loading marketplace data…")
def load_data():
    names = ["sellers", "listings", "search_events", "transactions"]
    return {n: pd.read_parquet(ROOT / "data" / f"{n}.parquet") for n in names}


@st.cache_data
def analyze(d):
    return (monthly_gmv_decomposition(d["transactions"]), seller_health(d["sellers"], d["listings"], d["transactions"]),
            price_elasticity(d["transactions"]), rank_ctr(d["search_events"]),
            search_funnel(d["search_events"], d["transactions"]), cohort_retention(d["transactions"]))


st.sidebar.title("Marketplace Intelligence")
st.sidebar.markdown("A decision platform tracing a GMV slowdown from marketplace mechanics through search discovery and a CUPED-powered ranking experiment.")
st.sidebar.markdown("[Executive memo](../docs/exec_memo.md)  \n[Product PRD](../docs/product_prd.md)  \n[Experiment readout](../docs/experiment_readout.md)")
d = load_data()
gmv, health, elasticity, ctr, funnel, retention = analyze(d)
category = st.sidebar.selectbox("Category", ["All"] + sorted(gmv.category.unique().tolist()))
st.title("Marketplace Growth & Pricing Intelligence Platform")
st.caption("Synthetic but internally related marketplace data • reproducible seed 42")

diagnose, investigate, validate = st.tabs(["Diagnose", "Investigate", "Validate"])
with diagnose:
    view = gmv if category == "All" else gmv[gmv.category == category]
    series = view.groupby("month", as_index=False).gmv.sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest monthly GMV", f"${series.iloc[-1].gmv:,.0f}", f"{series.iloc[-1].gmv/series.iloc[-4].gmv-1:.1%} vs 3 months ago")
    c2.metric("Healthy sellers", f"{(health.health_segment=='Healthy').mean():.1%}")
    most_elastic = elasticity.iloc[0]
    c3.metric("Most price-elastic category", most_elastic.category, f"ε = {most_elastic.elasticity:.2f}")
    st.plotly_chart(px.line(series, x="month", y="gmv", markers=True, title="Monthly GMV"), use_container_width=True)
    left, right = st.columns(2)
    seg = health.health_segment.value_counts().rename_axis("segment").reset_index(name="sellers")
    left.plotly_chart(px.pie(seg, names="segment", values="sellers", hole=.55, title="Seller health"), use_container_width=True)
    right.plotly_chart(px.bar(elasticity, x="category", y="elasticity", color="elasticity", title="Estimated log-log price elasticity"), use_container_width=True)

with investigate:
    st.plotly_chart(px.line(ctr, x="rank", y="ctr", markers=True, log_y=True, title="CTR contribution by rank (log scale)"), use_container_width=True)
    f = funnel if category == "All" else funnel[funnel.category == category]
    summary = f.groupby("rank_bucket", observed=True)[["clicks", "purchases"]].sum().reset_index()
    st.plotly_chart(px.bar(summary.melt("rank_bucket", var_name="stage", value_name="events"), x="rank_bucket", y="events", color="stage", barmode="group", title="Search funnel by rank bucket"), use_container_width=True)
    heat = go.Figure(go.Heatmap(z=retention.values, x=[f"M{i}" for i in retention.columns], y=retention.index, colorscale="Blues", text=(retention.values*100).round(1), texttemplate="%{text}%"))
    heat.update_layout(title="Buyer retention by first-purchase category", xaxis_title="Months since first purchase")
    st.plotly_chart(heat, use_container_width=True)

with validate:
    st.subheader("CUPED A/B test simulator")
    a, b, c = st.columns(3)
    n = a.slider("Total sample size", 10_000, 200_000, 40_000, 5_000)
    lift = b.slider("Relative effect size", 0.0, 0.20, 0.05, 0.01)
    corr = c.slider("Pre-period correlation strength", 0.0, 0.9, 0.5, 0.05)
    r = simulate_experiment(n=n, relative_lift=lift, correlation=corr)
    naive, cuped = r["naive"], r["cuped"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Naive p-value", f"{naive.p_value:.4f}", f"CI [{naive.ci_low:.3%}, {naive.ci_high:.3%}]")
    m2.metric("CUPED p-value", f"{cuped.p_value:.4f}", f"CI [{cuped.ci_low:.3%}, {cuped.ci_high:.3%}]")
    m3.metric("Variance reduction", f"{r['variance_reduction']:.1%}")
    st.caption(f"Power target: {required_sample_size(.08, max(lift, .001)):,} users per arm at 80% power and α=.05.")
    compare = pd.DataFrame({"method": ["Naive", "CUPED"], "effect": [naive.effect, cuped.effect],
                            "low": [naive.ci_low, cuped.ci_low], "high": [naive.ci_high, cuped.ci_high]})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=compare.effect, y=compare.method, mode="markers", marker_size=12,
        error_x=dict(type="data", symmetric=False, array=compare.high-compare.effect, arrayminus=compare.effect-compare.low)))
    fig.add_vline(x=0, line_dash="dash"); fig.update_layout(title="Absolute purchase-rate lift with 95% CI", xaxis_tickformat=".2%")
    st.plotly_chart(fig, use_container_width=True)
    st.json({k: round(v, 5) for k, v in r["guardrails"].items()})
