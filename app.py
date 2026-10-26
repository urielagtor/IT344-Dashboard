import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, auc


def auc_pr(y, p):
    """Same AUC-PR calculation as week3_fraud.py (trapezoid under the PR curve)."""
    prec, rec, _ = precision_recall_curve(y, p)
    return auc(rec, prec)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BENCH_AUCPR, BENCH_F1 = 0.478, 0.374          # benchmark.txt, Random Forest
FRAUD, LEGIT, MODEL, GREY = "#D55E00", "#0072B2", "#009E73", "#999999"

st.set_page_config(page_title="Fraud Alert Dashboard", layout="wide")
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-weight: 600; }
h1 { padding-bottom: 0; }
</style>
""", unsafe_allow_html=True)

st.title("Fraud Alert Dashboard")
st.subheader("How much fraud do we catch, and at what cost?")
st.markdown("**Goal:** Help fraud managers answer *\"With only ~4% of transactions being fraud, "
            "how much of it does our Random Forest catch, and how many legitimate customers do we bother "
            "to do it?\"* so they can decide what alert cutoff to use and how many analysts to staff "
            "for review next month.")


# ── Data ───────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv(os.path.join(DATA, "predictions.csv"), parse_dates=["transaction_datetime"])
    source = "Rebuilt Week 3 Random Forest (build_predictions.py)"
    sub_path = os.path.join(DATA, "submission.csv")
    if os.path.exists(sub_path):
        sub = pd.read_csv(sub_path).drop_duplicates("transaction_id")
        df = df.drop(columns="fraud_probability").merge(sub, on="transaction_id", how="inner")
        source = "Week 3 submission.csv"
    df["merchant_category"] = df["merchant_category"].fillna("unknown")
    df["channel"] = df["is_online"].map({"yes": "Online", "no": "In person"}).fillna("Unknown")
    df["week"] = df["transaction_datetime"].dt.to_period("W").dt.start_time
    return df, source


def wilson(k, n, z=1.96):
    """95% interval for a proportion k/n (works for small weekly counts)."""
    n = np.asarray(n, float)
    k = np.asarray(k, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        den = 1 + z**2 / n
        mid = (p + z**2 / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return mid - half, mid + half


df_all, score_source = load()

# ── Sidebar: filters + notes ───────────────────────────────────────────────
cutoff = st.slider("Alert cutoff: flag a transaction if its fraud score is at least",
                   0.05, 0.90, 0.20, 0.05,
                   help="Week 3 used 0.20. Lower = catch more fraud but more false alarms. "
                        "Higher = fewer false alarms but more fraud slips through.")
st.sidebar.header("Filters")
cats = sorted(df_all["merchant_category"].unique())
pick_cats = st.sidebar.multiselect("Merchant category", cats, default=cats)
pick_chan = st.sidebar.multiselect("Channel", ["Online", "In person", "Unknown"],
                                   default=["Online", "In person", "Unknown"])
dmin, dmax = df_all["transaction_datetime"].min().date(), df_all["transaction_datetime"].max().date()
d_range = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)

df = df_all[df_all["merchant_category"].isin(pick_cats) & df_all["channel"].isin(pick_chan)]
if isinstance(d_range, tuple) and len(d_range) == 2:
    in_range = df["transaction_datetime"].dt.date.between(*d_range)
    # keep rows with a missing timestamp unless the user narrowed the dates
    if d_range != (dmin, dmax):
        df = df[in_range]
if df.empty or df["is_fraud"].sum() == 0:
    st.warning("No fraud cases in this selection. Widen the filters.")
    st.stop()

df = df.assign(flag=(df["fraud_probability"] >= cutoff).astype(int))
tp = int(((df.flag == 1) & (df.is_fraud == 1)).sum())
fp = int(((df.flag == 1) & (df.is_fraud == 0)).sum())
fn = int(((df.flag == 0) & (df.is_fraud == 1)).sum())
tn = int(((df.flag == 0) & (df.is_fraud == 0)).sum())
recall = tp / (tp + fn)
precision = tp / (tp + fp) if tp + fp else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
aucpr = auc_pr(df["is_fraud"], df["fraud_probability"])
fpr_per_1k = 1000 * fp / (fp + tn)
usd_caught = df.loc[(df.flag == 1) & (df.is_fraud == 1), "transaction_amount"].sum()
usd_total = df.loc[df.is_fraud == 1, "transaction_amount"].sum()
top = df.nlargest(max(1, len(df) // 100), "fraud_probability")   # highest-risk 1%
top_hit = top["is_fraud"].mean()

# Notes panel — always describes the full eval set, not the filtered view
full = df_all.assign(flag=(df_all["fraud_probability"] >= 0.20).astype(int))
f_tp = ((full.flag == 1) & (full.is_fraud == 1)).sum()
f_p = f_tp / full.flag.sum()
f_r = f_tp / full.is_fraud.sum()
st.sidebar.divider()
with st.sidebar.expander("Notes: data, model, metrics", expanded=True):
    st.markdown(f"""
**Data source:** Course fraud dataset — `transactions_train.csv` (train),
`transactions_eval.csv` + `evaluation_labels.csv` (test), joined to
`customers.csv` and `merchants.csv`.

**Date range (test):** {dmin:%b %d, %Y} – {dmax:%b %d, %Y}
({len(df_all):,} transactions, {df_all.is_fraud.mean():.1%} fraud;
{df_all.transaction_datetime.isna().sum():,} have no timestamp and are left out of the weekly charts).

**Model:** Random Forest, 100 trees, `class_weight='balanced_subsample'`, random_state 42 (matches the Week 3 reported results).
10 features (amount, distance, hours since last txn, ratio to median,
txns in 24h, account age, online, card present, category, international).
Missing values filled with 0. Date/time not used yet.

**Scores shown:** {score_source}

**Key metrics (full test set, cutoff 0.20):** AUC-PR {auc_pr(full.is_fraud, full.fraud_probability):.3f}
· precision {f_p:.3f} · recall {f_r:.3f} · F1 {2*f_p*f_r/(f_p+f_r):.3f}

**Benchmark (benchmark.txt):** RF AUC-PR {BENCH_AUCPR}, F1 {BENCH_F1}
""")

# ── Header ─────────────────────────────────────────────────────────────────
st.caption(f"Showing {len(df):,} test transactions · alert cutoff {cutoff:.2f} · filters in the sidebar")

# ── 1. KPI cards ───────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Fraud caught", f"{recall:.0%}",
          help=f"{tp:,} of {tp+fn:,} fraud cases were flagged.")
k1.caption(f"Catches about **{recall*10:.1f} of every 10** fraud cases")
k2.metric("Alerts that are real fraud", f"{precision:.0%}",
          help=f"{tp:,} of {tp+fp:,} alerts were actual fraud.")
k2.caption(f"About **{precision*10:.1f} of every 10** alerts are worth a call")
k3.metric("False alarms", f"{fpr_per_1k:.0f} per 1,000",
          help=f"{fp:,} legitimate transactions were flagged.")
k3.caption("legitimate customers bothered")
k4.metric("Fraud $ stopped", f"${usd_caught/1e6:.1f}M",
          help=f"Out of ${usd_total/1e6:.1f}M in fraudulent transactions in view.")
k4.caption(f"{usd_caught/usd_total:.0%} of fraud dollars in view")
k5.metric("Ranking quality", f"{aucpr:.3f}", f"{aucpr-BENCH_AUCPR:+.3f} vs benchmark",
          help="AUC-PR: area under the precision-recall curve. Measures how well fraud is ranked above "
               "legitimate transactions at every cutoff, so it is not affected by the cutoff you pick. "
               "Higher is better; benchmark Random Forest = 0.478.")
k5.caption(f"AUC-PR · top 1% riskiest are right **{top_hit*10:.1f} of 10** times")

st.divider()

# ── 2 & 3. Weekly: predicted vs actual, and catch rate with error band ────
wk = (df.dropna(subset=["week"])
        .assign(caught=lambda d: d.flag * d.is_fraud)
        .groupby("week")
        .agg(actual=("is_fraud", "sum"), flagged=("flag", "sum"),
             caught=("caught", "sum"), n=("flag", "size"))
        .reset_index())
wk = wk[wk["n"] >= 0.5 * wk["n"].median()]          # drop partial first/last weeks
wk["recall"] = wk["caught"] / wk["actual"]
wk["lo"], wk["hi"] = wilson(wk["caught"], wk["actual"])
mae_weekly = (wk["flagged"] - wk["actual"]).abs().mean()

c1, c2 = st.columns(2)
with c1:
    st.subheader("Predicted vs actual fraud, by week")
    fig = go.Figure()
    fig.add_scatter(x=wk.week, y=wk.actual, name="Actual fraud", mode="lines+markers",
                    line=dict(color=FRAUD, width=3))
    fig.add_scatter(x=wk.week, y=wk.flagged, name="Model alerts", mode="lines+markers",
                    line=dict(color=LEGIT, width=2))
    fig.add_scatter(x=wk.week, y=wk.caught, name="Alerts that were fraud", mode="lines",
                    line=dict(color=MODEL, width=2), fill="tozeroy", fillcolor="rgba(0,158,115,0.12)")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2), yaxis_title="Transactions per week")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Weekly alert volume is typically off from actual fraud by **~{mae_weekly:,.0f} "
               f"transactions per week** (mean absolute difference). The gap between the orange "
               f"and green lines is fraud that slipped through.")

with c2:
    st.subheader("Weekly catch rate with 95% band")
    fig = go.Figure()
    fig.add_scatter(x=pd.concat([wk.week, wk.week[::-1]]),
                    y=pd.concat([wk.hi, wk.lo[::-1]]), fill="toself",
                    fillcolor="rgba(0,158,115,0.18)", line=dict(width=0),
                    name="95% band", hoverinfo="skip")
    fig.add_scatter(x=wk.week, y=wk.recall, name="Share of fraud caught", mode="lines+markers",
                    line=dict(color=MODEL, width=3))
    fig.add_hline(y=recall, line_dash="dot", line_color=GREY,
                  annotation_text=f"overall {recall:.0%}", annotation_position="top left")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      yaxis=dict(tickformat=".0%", range=[0, 1], title="Fraud caught"),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Week to week the catch rate moves between **{wk.recall.min():.0%} and "
               f"{wk.recall.max():.0%}**. The band shows how much of that is just small weekly counts.")

# ── 4. Confusion summary  &  5. By merchant category ─────────────────────
c3, c4 = st.columns([2, 3])
with c3:
    st.subheader("Where the model is right and wrong")
    z = [[tp, fn], [fp, tn]]
    text = [[f"<b>{tp:,}</b><br>fraud caught", f"<b>{fn:,}</b><br>fraud missed"],
            [f"<b>{fp:,}</b><br>false alarms", f"<b>{tn:,}</b><br>correctly cleared"]]
    shade = [[1.0, 0.55], [0.55, 0.15]]
    fig = go.Figure(go.Heatmap(z=shade, text=text, texttemplate="%{text}", showscale=False,
                               colorscale=[[0, "#f2f2f2"], [1, "#9ecae1"]],
                               x=["Model: fraud", "Model: OK"], y=["Actually fraud", "Actually legit"],
                               hoverinfo="skip", xgap=3, ygap=3))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed"), font=dict(size=14))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"For every fraud caught there are **{fn/max(tp,1):.1f} missed** and "
               f"**{fp/max(tp,1):.1f} false alarms**.")

with c4:
    st.subheader("Predicted vs actual fraud rate, by merchant category")
    cat = (df.groupby("merchant_category")
             .agg(n=("flag", "size"), actual=("is_fraud", "mean"), predicted=("flag", "mean"),
                  fraud=("is_fraud", "sum"))
             .query("n >= 200 and fraud >= 10")
             .sort_values("actual", ascending=False).reset_index())
    caught = df[df.is_fraud == 1].groupby("merchant_category")["flag"].mean()
    cat["recall"] = cat["merchant_category"].map(caught)
    fig = go.Figure()
    fig.add_bar(x=cat.merchant_category, y=cat.actual, name="Actual fraud rate", marker_color=FRAUD)
    fig.add_bar(x=cat.merchant_category, y=cat.predicted, name="Model alert rate", marker_color=LEGIT,
                customdata=cat.recall,
                hovertemplate="%{x}<br>alert rate %{y:.1%}<br>fraud caught %{customdata:.0%}<extra></extra>")
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(tickformat=".0%", title="Share of transactions"),
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    if cat.empty:
        st.caption("Too few transactions in this selection to compare categories (need 200+ and 10+ fraud).")
    else:
        best, worst = cat.loc[cat.recall.idxmax()], cat.loc[cat.recall.idxmin()]
        st.caption(f"Catches the most fraud in **{best.merchant_category}** ({best.recall:.0%}) and the least "
                   f"in **{worst.merchant_category}** ({worst.recall:.0%}). Categories with <200 transactions hidden.")

# ── 6. Review queue: prediction next to outcome ───────────────────────────
st.subheader("Highest-risk transactions: prediction vs actual outcome")
show_n = st.slider("Rows to show", 10, 200, 25, 5)
queue = df.nlargest(show_n, "fraud_probability")[
    ["transaction_id", "transaction_datetime", "merchant_category", "channel",
     "transaction_amount", "fraud_probability", "flag", "is_fraud"]].copy()
queue["Prediction"] = np.where(queue.flag == 1, "Fraud", "OK")
queue["Actual"] = np.where(queue.is_fraud == 1, "Fraud", "Legit")
queue["Result"] = np.select(
    [(queue.flag == 1) & (queue.is_fraud == 1), (queue.flag == 1) & (queue.is_fraud == 0),
     (queue.flag == 0) & (queue.is_fraud == 1)],
    ["✅ caught", "⚠️ false alarm", "❌ missed"], "✅ cleared")
st.dataframe(
    queue.drop(columns=["flag", "is_fraud"]).rename(columns={
        "transaction_id": "Transaction", "transaction_datetime": "When",
        "merchant_category": "Category", "channel": "Channel",
        "transaction_amount": "Amount", "fraud_probability": "Fraud score"}),
    width="stretch", hide_index=True,
    column_config={"Amount": st.column_config.NumberColumn(format="$%.2f"),
                   "Fraud score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
st.caption(f"In this list the model is right **{(queue.Result.str.startswith('✅')).mean()*10:.1f} times out of 10**.")
