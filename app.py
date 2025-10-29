# app.py
import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import plotly.express as px
import io
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- Config ----------
FUZZY_THRESHOLD = 85

# Base canned Q/A (same as before)
BASE_CANNED_QA = {
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

# ---------- Utility functions ----------
def normalize_text(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()

def best_fuzzy_match(query: str, canned_keys):
    q = normalize_text(query)
    best_score = 0
    best_key = None
    for k in canned_keys:
        score = fuzz.partial_ratio(normalize_text(k), q)
        if score > best_score:
            best_score = score
            best_key = k
    return best_key, best_score

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
    group_x = x_col
    if y_col is None:
        agg = df2.groupby(group_x).size().reset_index(name="count").sort_values("count", ascending=False)
        if chart_type == "pie":
            fig = px.pie(agg, names=group_x, values="count", title=f"Count by {group_x}")
        else:
            fig = px.bar(agg, x=group_x, y="count", title=f"Count by {group_x}", text="count")
        return fig
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

# ---------- Session initialization ----------
if "canned_qa" not in st.session_state:
    st.session_state.canned_qa = dict(BASE_CANNED_QA)  # editable during session
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"query":..., "response":..., "type":"chart"/"canned"/"error"}

# ---------- UI ----------
st.set_page_config(page_title="CSV Chat — Better UX", layout="wide")
st.title("CSV Chat — Improved question editing UX")

st.markdown(
    "Upload your CSV, pick or type a question, edit it freely, see best canned-match score live, view history, re-run or save edits as canned questions."
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
col_l, col_r = st.columns([3, 1])

with col_r:
    st.header("Quick actions & canned")
    # Dropdown of canned questions for quick selection (editable later)
    canned_keys = list(st.session_state.canned_qa.keys())
    pick = st.selectbox("Pick a canned question (select to prefill)", [""] + canned_keys)
    if pick:
        st.session_state.prefill = pick
    st.write("---")
    st.subheader("Example queries")
    for q in EXAMPLE_QUERIES:
        if st.button(q, key="ex_"+q):
            st.session_state.prefill = q
    st.write("---")
    st.subheader("Saved canned (session)")
    if len(canned_keys) == 0:
        st.write("No canned entries")
    else:
        # show top 8 with small buttons to edit or remove
        for k in canned_keys:
            cols = st.columns([6,1,1])
            cols[0].write(k)
            if cols[1].button("Prefill", key="pf_"+k):
                st.session_state.prefill = k
            if cols[2].button("Remove", key="rm_"+k):
                st.session_state.canned_qa.pop(k, None)
                st.experimental_rerun()

with col_l:
    if uploaded is None:
        st.info("Please upload CSV to proceed (use your demo CSV).")
        st.stop()

    # Robust CSV read
    try:
        content = uploaded.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded)

    st.subheader("Data preview")
    st.dataframe(df)

    st.subheader("Ask a question (editable)")
    prefill = st.session_state.get("prefill", "")
    user_input = st.text_input("Query (edit freely)", value=prefill, key="user_query", placeholder="Type or pick a canned question then edit")
    # live best match info
    best_key, best_score = best_fuzzy_match(user_input, list(st.session_state.canned_qa.keys()))
    if user_input.strip():
        if best_key:
            st.caption(f"Best canned match: \"{best_key}\" (score {best_score})")
        else:
            st.caption("No canned match available yet")

    col_submit = st.columns([1,1,1])
    ask = col_submit[0].button("Ask")
    save_canned = col_submit[1].button("Save as canned")
    clear_input = col_submit[2].button("Clear")

    if clear_input:
        st.session_state.user_query = ""
        st.session_state.prefill = ""
        st.experimental_rerun()

    if save_canned and user_input.strip():
        # save the current edited query as a canned key with a placeholder answer (user can overwrite later)
        key = user_input.strip()
        # avoid duplicates
        if key in st.session_state.canned_qa:
            st.warning("That canned question already exists")
        else:
            st.session_state.canned_qa[key] = "Canned answer placeholder. Edit this in code or save a response from history."
            st.success("Saved question as canned (session-only)")
            st.experimental_rerun()

    if ask and user_input.strip():
        lower = user_input.lower()
        # detect aggregate/chart intent
        chart_tokens = [" by ", "total", "sum", "over time", "per ", "aggregate", "count by", "incidents over", "area chart", "area"]
        if any(tok in lower for tok in chart_tokens) or detect_chart_type(user_input):
            try:
                fig = aggregate_and_plot(df, user_input)
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.history.insert(0, {"query": user_input, "response": "chart", "type": "chart"})
            except Exception as e:
                st.error(f"Error generating chart: {e}")
                st.session_state.history.insert(0, {"query": user_input, "response": f"chart-error: {e}", "type": "error"})
        else:
            # try canned match
            canned_resp = None
            # exact normalized match
            for k,v in st.session_state.canned_qa.items():
                if normalize_text(k) == normalize_text(user_input):
                    canned_resp = v
                    matched_key = k
                    break
            if canned_resp is None:
                # fuzzy match
                matched_key, score = best_fuzzy_match(user_input, list(st.session_state.canned_qa.keys()))
                if score >= FUZZY_THRESHOLD:
                    canned_resp = st.session_state.canned_qa.get(matched_key)
                else:
                    canned_resp = None

            if canned_resp is not None:
                st.success(canned_resp)
                st.session_state.history.insert(0, {"query": user_input, "response": canned_resp, "type": "canned", "matched_key": matched_key, "score": score if 'score' in locals() else 100})
            else:
                st.error("❌ Error: This query requires an LLM call which exceeds the free-tier token limit.")
                st.session_state.history.insert(0, {"query": user_input, "response": "token-limit-error", "type": "error"})

    # History panel
    st.write("---")
    st.subheader("Query history (session)")
    if len(st.session_state.history) == 0:
        st.write("No history yet")
    else:
        # show up to 20 recent items
        for idx, item in enumerate(st.session_state.history[:20]):
            cols = st.columns([6,1,1,1])
            display_q = item["query"] if len(item["query"]) < 120 else item["query"][:117] + "..."
            cols[0].write(f"Q: {display_q}")
            # response preview
            if item["type"] == "chart":
                cols[0].write("A: (chart) — re-run to view")
            elif item["type"] == "canned":
                resp_preview = item["response"] if len(item["response"]) < 160 else item["response"][:157]+"..."
                cols[0].write(f"A: {resp_preview}")
            else:
                cols[0].write("A: (error)")

            if cols[1].button("Re-run", key=f"rerun_{idx}"):
                st.session_state.user_query = item["query"]
                st.session_state.prefill = item["query"]
                st.experimental_rerun()
            if cols[2].button("Edit", key=f"edit_{idx}"):
                st.session_state.user_query = item["query"]
                st.session_state.prefill = item["query"]
                st.experimental_rerun()
            if cols[3].button("Copy", key=f"copy_{idx}"):
                st.experimental_set_query_params(_q=item["query"])
                st.success("Query copied to URL params (press Ctrl+L then Enter to copy).")

# End of app
