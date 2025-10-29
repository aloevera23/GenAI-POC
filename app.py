# app.py
import streamlit as st
import pandas as pd
import io
from fuzzywuzzy import fuzz
import plotly.express as px

# ---------- Config ----------
FUZZY_THRESHOLD = 85

# ---------- Canned Q/A (original + 5 additional) ----------
CANNED_QA = {
    "what is the total number of incidents": "Total incidents: 21.",
    "how many incidents by severity": "Severity counts: Low: 10; Medium: 5; High: 4; Critical: 2.",
    "which area has the most incidents": "Area counts: Logistics: 7; Control Room: 4; Drilling: 4; Maintenance: 2; Accommodation: 2; Production: 1. Top area: Logistics (7 incidents).",
    "what is the total damage cost and average damage cost per incident": "Total Damage Cost: 2,937,133. Average Damage Cost per incident: ~139,864.",
    "how many days lost in total and average days lost": "Total Days Lost: 34. Average Days Lost per incident: ~1.62.",
    "top 3 root causes by frequency": "Top 3 Root Causes: Environmental (7), Mechanical (6), Procedural (3).",
    "which incident types are most common": "Incident Type counts: Environmental: 5; Other: 4; Equipment Failure: 4; Slip/Trip: 3; Human Error: 2; Process Safety: 2; Fire/Explosion: 1.",
    "highest damage cost incident": "Highest Damage Cost: ID 4 — 465,817 — Slip/Trip in Drilling — Severity Low.",
    "count incidents by personnel involved groups": "Personnel buckets: 1-3: 5 incidents; 4-6: 8 incidents; 7-9: 8 incidents.",
    "number of critical incidents and their ids": "Critical incidents: 2. IDs: 19, 20.",
    # 5 additional questions (3 non-plot, 2 plot)
    "which incidents have severity high and what are their ids": "High severity incidents: IDs 1, 6, 7, 14.",
    "what are the ids of incidents with damage cost over 400000": "IDs with Damage Cost > 400,000: 4, 5, 9, 15.",
    "how many incidents caused more than 2 days lost": "Incidents with Days Lost > 2: 8 incidents (IDs 4, 5, 9, 10, 12, 14, 16, 17).",
    "show incidents by area (aggregate bar chart)": "Returning an aggregate bar chart showing incident counts by Area.",
    "area chart: total damage cost by area (aggregate area chart)": "Returning an area chart showing total Damage Cost aggregated by Area."
}

# Examples should list all canned questions (combined)
EXAMPLES = list(CANNED_QA.keys())

# ---------- Helpers ----------
def normalize_text(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()

def find_canned_response(user_query: str):
    q = normalize_text(user_query)
    # exact normalized match
    for k, v in CANNED_QA.items():
        if normalize_text(k) == q:
            return v, k, 100
    # fuzzy match
    best_score = 0
    best_key = None
    for k in CANNED_QA:
        score = fuzz.partial_ratio(normalize_text(k), q)
        if score > best_score:
            best_score = score
            best_key = k
    if best_score >= FUZZY_THRESHOLD:
        return CANNED_QA[best_key], best_key, best_score
    return None, None, 0

def detect_chart_intent(query: str):
    q = query.lower()
    chart_tokens = [" by ", " total ", " sum ", "over time", "per ", "aggregate", "count by", "incidents over", "area chart", "area", "chart", "plot", "show"]
    return any(tok in q for tok in chart_tokens)

def ensure_datetime(df: pd.DataFrame, date_col="Date"):
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, errors="coerce")
    return df

def choose_columns_for_agg(query: str, df: pd.DataFrame):
    q = query.lower()
    if "by area" in q or " area " in q:
        x = "Area"
    elif "by severity" in q or " severity " in q:
        x = "Severity"
    elif "incident type" in q or "by incident type" in q:
        x = "Incident Type"
    elif "month" in q or "over time" in q or "time" in q:
        x = "Date"
    else:
        cats = [c for c in df.columns if df[c].dtype == object]
        x = cats[0] if cats else df.columns[0]
    if "damage" in q or "damage cost" in q or "sum damage" in q or "total damage" in q:
        y = "Damage Cost"
    elif "days" in q or "days lost" in q:
        y = "Days Lost"
    elif "count" in q or "number of" in q or "incidents" in q:
        y = None
    else:
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        y = nums[0] if nums else None
    return x, y

def aggregate_plot(df: pd.DataFrame, query: str):
    df2 = ensure_datetime(df, "Date")
    x_col, y_col = choose_columns_for_agg(query, df2)
    # Date/time over time handling
    if x_col == "Date":
        df2["month"] = df2["Date"].dt.to_period("M").astype(str)
        group_x = "month"
        if y_col is None:
            agg = df2.groupby(group_x).size().reset_index(name="count")
            fig = px.line(agg, x=group_x, y="count", title="Incidents Over Time (monthly)")
            fig.update_layout(xaxis_tickangle=-45)
            return fig
        else:
            agg = df2.groupby(group_x)[y_col].sum().reset_index().sort_values(group_x)
            fig = px.line(agg, x=group_x, y=y_col, title=f"{y_col} over time (monthly)")
            fig.update_layout(xaxis_tickangle=-45)
            return fig
    # Categorical x
    group_x = x_col
    if y_col is None:
        agg = df2.groupby(group_x).size().reset_index(name="count").sort_values("count", ascending=False)
        fig = px.bar(agg, x=group_x, y="count", title=f"Count by {group_x}", text="count")
        return fig
    # Numeric y aggregation
    agg = df2.groupby(group_x)[y_col].sum().reset_index().sort_values(y_col, ascending=False)
    ql = query.lower()
    if "area" in ql or "area chart" in ql:
        fig = px.area(agg, x=group_x, y=y_col, title=f"Total {y_col} by {group_x}")
    elif "pie" in ql or "donut" in ql:
        fig = px.pie(agg, names=group_x, values=y_col, title=f"Total {y_col} by {group_x}")
    elif "line" in ql or "over time" in ql:
        fig = px.line(agg, x=group_x, y=y_col, title=f"Total {y_col} by {group_x}")
    else:
        fig = px.bar(agg, x=group_x, y=y_col, title=f"Total {y_col} by {group_x}", text=y_col)
    return fig

# ---------- Streamlit UI ----------
st.set_page_config(page_title="CSV Chat — simple", layout="wide")
st.title("CSV Chat — simple behavior (canned + plots)")

# If no upload, show only this message
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if not uploaded:
    st.info("Please upload the CSV.")
    st.stop()

# read csv
try:
    content = uploaded.read()
    df = pd.read_csv(io.BytesIO(content))
except Exception:
    uploaded.seek(0)
    df = pd.read_csv(uploaded)

st.subheader("Data preview")
st.dataframe(df)

st.subheader("Ask a question")
query = st.text_input("Type your question here (try an example)", key="q_input")
if st.button("Ask") and query:
    is_plot = detect_chart_intent(query)
    canned_answer, canned_key, score = find_canned_response(query)
    # Behavior rules:
    # - If plot intent: show plot. If also canned matched -> show text below.
    # - If no plot intent:
    #    - if canned matched -> show canned text
    #    - else -> show token-limit error
    if is_plot:
        try:
            fig = aggregate_plot(df, query)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating plot: {e}")
        if canned_answer:
            st.write("")  # spacing
            st.success(canned_answer)
    else:
        if canned_answer:
            st.success(canned_answer)
        else:
            st.error("❌ Error: This query requires an LLM call which exceeds the free-tier token limit.")

# show all canned questions as examples
st.write("Example queries:")
for ex in EXAMPLES:
    st.write("- " + ex)
