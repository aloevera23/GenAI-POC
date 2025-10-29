# app.py
import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import plotly.express as px
import io
import os
from dotenv import load_dotenv

# Load .env if present (not required for canned/demo mode)
load_dotenv()

# --- Configuration ---
FUZZY_THRESHOLD = 85

# --- Canned Q/A derived from the CSV you provided ---
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
    "number of critical incidents and their ids": "Critical incidents: 2. IDs: 19, 20."
}

EXAMPLE_QUERIES = [
    "What is the total number of incidents",
    "How many incidents by Severity",
    "Which Area has the most incidents",
    "What is the total Damage Cost and average Damage Cost per incident",
    "How many days lost in total and average days lost",
    "Top 3 Root Causes by frequency",
    "Which Incident Types are most common",
    "Highest Damage Cost incident",
    "Count incidents by Personnel Involved groups (e.g., 1-3, 4-6, 7-9)",
    "Number of Critical incidents and their IDs",
    "Show incidents by Area (aggregate bar chart)",
    "Sum Damage Cost by Severity (aggregate bar chart)",
    "Incidents over time (monthly line chart)",
    "Area chart: total Damage Cost by Area (aggregate area chart)"
]

# --- Helpers ---

def normalize_text(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()

def find_canned_response(user_query: str):
    q = normalize_text(user_query)
    # exact normalized match
    for k, v in CANNED_QA.items():
        if normalize_text(k) == q:
            return v
    # fuzzy match against canned keys
    best_score = 0
    best_key = None
    for k in CANNED_QA.keys():
        score = fuzz.partial_ratio(normalize_text(k), q)
        if score > best_score:
            best_score = score
            best_key = k
    if best_score >= FUZZY_THRESHOLD:
        return CANNED_QA[best_key]
    return None

def detect_chart_type(query: str):
    q = query.lower()
    chart_keywords = {
        "bar": ["bar", "bar chart", "count by", "by area", "by severity"],
        "line": ["over time", "trend", "line chart", "monthly"],
        "pie": ["pie", "donut", "share", "proportion"],
        "scatter": ["scatter", "scatterplot", "correlation"],
        "area": ["area chart", "area", "area chart:"]
    }
    for chart, keywords in chart_keywords.items():
        for word in keywords:
            if word in q:
                return chart
    return None

def extract_columns_for_aggregate(query: str, df: pd.DataFrame):
    q = query.lower()
    # X axis candidate
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
    # Y axis candidate
    if "damage" in q or "damage cost" in q or "sum damage" in q or "total damage" in q:
        y = "Damage Cost"
    elif "days" in q or "days lost" in q:
        y = "Days Lost"
    elif "count" in q or "number of" in q or "incidents" in q:
        y = None  # count
    else:
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        y = nums[0] if nums else None
    return x, y

def ensure_datetime(df: pd.DataFrame, date_col="Date"):
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, errors="coerce")
    return df

def aggregate_and_plot(df: pd.DataFrame, query: str):
    chart_type = detect_chart_type(query)
    x_col, y_col = extract_columns_for_aggregate(query, df)
    df2 = df.copy()
    df2 = ensure_datetime(df2, "Date")
    # If x is Date and user asked "over time", aggregate by month
    if x_col == "Date":
        df2["month"] = df2["Date"].dt.to_period("M").astype(str)
        group_x = "month"
        if y_col is None:
            agg = df3 = df2.groupby(group_x).size().reset_index(name="count")
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
        if chart_type == "pie":
            fig = px.pie(agg, names=group_x, values="count", title=f"Count by {group_x}")
        else:
            fig = px.bar(agg, x=group_x, y="count", title=f"Count by {group_x}", text="count")
        return fig

    # numeric y aggregation
    agg = df2.groupby(group_x)[y_col].sum().reset_index().sort_values(y_col, ascending=False)
    if chart_type == "area":
        fig = px.area(agg, x=group_x, y=y_col, title=f"Total {y_col} by {group_x}")
    elif chart_type == "pie":
        fig = px.pie(agg, names=group_x, values=y_col, title=f"Total {y_col} by {group_x}")
    elif chart_type == "line":
        fig = px.line(agg, x=group_x, y=y_col, title=f"Total {y_col} by {group_x}")
    else:
        fig = px.bar(agg, x=group_x, y=y_col, title=f"Total {y_col} by {group_x}", text=y_col)
    return fig

# --- Streamlit UI ---

st.set_page_config(page_title="CSV Chat Demo (canned)", layout="wide")
st.title("CSV Chat Demo — canned responses + Plotly aggregates")

st.markdown("Upload the CSV (use the provided mock_oil_rig_incidents CSV). Example queries are shown on the right. Charts support aggregation including area charts.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

col_left, col_right = st.columns([2, 1])

with col_right:
    st.header("Example questions")
    for q in EXAMPLE_QUERIES:
        if st.button(q, key=q):
            st.session_state["prefill"] = q
    st.write("---")
    st.caption("Example behavior: clicking a question pre-fills the input; charts are computed with pandas groupby and Plotly; unmatched free-text returns a token-limit simulated error.")

with col_left:
    if "prefill" not in st.session_state:
        st.session_state["prefill"] = ""
    if uploaded is None:
        st.info("Please upload the CSV (mock_oil_rig_incidents CSV).")
    else:
        # robust CSV read
        try:
            content = uploaded.read()
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded)
        st.subheader("Data preview")
        st.dataframe(df)
        st.subheader("Ask a question")
        user_input = st.text_input("Query", value=st.session_state.get("prefill", ""), key="query_input")
        submit = st.button("Ask")

        if submit and user_input:
            st.session_state["prefill"] = ""
            lower = user_input.lower()
            # treat chart/aggregate intent if tokens appear
            chart_tokens = [" by ", "total", "sum", "over time", "per ", "aggregate", "count by", "incidents over", "area chart", "area"]
            if any(tok in lower for tok in chart_tokens) or detect_chart_type(user_input):
                try:
                    fig = aggregate_and_plot(df, user_input)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating aggregate chart: {e}")
            else:
                canned = find_canned_response(user_input)
                if canned is not None:
                    st.success(canned)
                else:
                    st.error("❌ Error: This query requires an LLM call which exceeds the free-tier token limit. Please try one of the example questions.")
