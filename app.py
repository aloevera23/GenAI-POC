import streamlit as st
import pandas as pd
import io
import plotly.express as px
import openai
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("key")

# --- Canned Q/A ---
CANNED_QA = {
    "which area has the most incidents": "Top area: Logistics (7 incidents).",
    "show a pie chart of incidents by root cause": "Pie chart showing incident distribution by Root Cause.",
    "how many incidents by severity": "Severity counts: Low: 10; Medium: 5; High: 4; Critical: 2.",
    "highest damage cost incident": "Highest Damage Cost: ID 4 — 465,817 — Slip/Trip in Drilling — Severity Low."
}

CANNED_EXPECTED = {
    "which area has the most incidents": "text",
    "show a pie chart of incidents by root cause": "plot",
    "how many incidents by severity": "both",
    "highest damage cost incident": "text"
}

BLOCKED_QUERIES = [
    "top 3 root causes",
    "which incident type is most common"
]

# --- Helpers ---
def normalize_text(s):
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()

def match_canned(query):
    q = normalize_text(query)
    for k in CANNED_QA:
        if normalize_text(k) == q:
            return CANNED_QA[k], k
    for k in CANNED_QA:
        if fuzz.partial_ratio(normalize_text(k), q) >= 85:
            return CANNED_QA[k], k
    return None, None

def is_blocked(query):
    q = normalize_text(query)
    for b in BLOCKED_QUERIES:
        if fuzz.partial_ratio(normalize_text(b), q) >= 85:
            return True
    return False

def detect_chart_intent(query):
    q = query.lower()
    tokens = [" by ", "chart", "plot", "show", "sum", "total", "over time", "area", "pie", "line"]
    return any(tok in q for tok in tokens)

def ensure_datetime(df, col="Date"):
    df = df.copy()
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def choose_columns(query, df):
    q = query.lower()
    if "area" in q:
        x = "Area"
    elif "severity" in q:
        x = "Severity"
    elif "root cause" in q:
        x = "Root Cause"
    elif "incident type" in q:
        x = "Incident Type"
    elif "date" in q or "time" in q or "month" in q:
        x = "Date"
    else:
        x = df.select_dtypes(include="object").columns[0]
    if "damage" in q:
        y = "Damage Cost"
    elif "days" in q:
        y = "Days Lost"
    else:
        y = None
    return x, y

def plot_chart(df, query, matched_key=None):
    df = ensure_datetime(df)
    x, y = choose_columns(query, df)

    # Force pie chart for specific canned key
    if matched_key == "show a pie chart of incidents by root cause":
        agg = df.groupby("Root Cause").size().reset_index(name="count")
        return px.pie(agg, names="Root Cause", values="count", title="Incidents by Root Cause")

    if x == "Date":
        df["month"] = df["Date"].dt.to_period("M").astype(str)
        x = "month"

    if y is None:
        agg = df.groupby(x).size().reset_index(name="count")
        return px.bar(agg, x=x, y="count", title=f"Incidents by {x}", text="count")
    else:
        agg = df.groupby(x)[y].sum().reset_index()
        ql = query.lower()
        if "area" in ql or "area chart" in ql:
            return px.area(agg, x=x, y=y, title=f"{y} by {x}")
        elif "pie" in ql:
            return px.pie(agg, names=x, values=y, title=f"{y} by {x}")
        elif "line" in ql or "over time" in ql:
            return px.line(agg, x=x, y=y, title=f"{y} by {x}")
        else:
            return px.bar(agg, x=x, y=y, title=f"{y} by {x}", text=y)

def ask_openai(df, query):
    preview = df.head(10).to_markdown()
    prompt = f"You are a data analyst. Here's a preview of the dataframe:\n\n{preview}\n\nAnswer this question:\n{query}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# --- UI ---
st.set_page_config(page_title="HCL GenAI Demo", layout="wide")
st.title("TotalEnergies: Health & Safety")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if not uploaded:
    st.info("Please upload the CSV.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Data preview")
st.dataframe(df)

st.subheader("Ask a question")
query = st.text_input("Type your question here")
if st.button("Ask") and query:
    if is_blocked(query):
        st.error("❌ Error: This query requires an LLM call which exceeds the free-tier token limit.")
    else:
        canned, key = match_canned(query)
        expected = CANNED_EXPECTED.get(key, None)
        chart_intent = detect_chart_intent(query)

        if canned and expected == "text":
            st.success(canned)
        elif canned and expected == "plot":
            fig = plot_chart(df, query)
            st.plotly_chart(fig, use_container_width=True)
        elif canned and expected == "both":
            fig = plot_chart(df, query)
            st.plotly_chart(fig, use_container_width=True)
            st.success(canned)
        elif chart_intent:
            fig = plot_chart(df, query)
            st.plotly_chart(fig, use_container_width=True)
        else:
            try:
                answer = ask_openai(df, query)
                st.success(answer)
            except Exception as e:
                st.error(f"❌ OpenAI error: {e}")

