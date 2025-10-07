import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from pandasai.pandasai import PandasAI
from pandasai.llms import OpenAI
from fuzzywuzzy import fuzz
import plotly.express as px
import random

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('key')  # .env should contain: key=your_openai_api_key

# Detect chart type using fuzzy matching or verbs
def detect_chart_type(query):
    chart_keywords = {
        "bar": ["bar", "barplot", "bar chart"],
        "line": ["line", "trend", "line chart"],
        "pie": ["pie", "donut", "pie chart"],
        "scatter": ["scatter", "scatterplot", "dot chart"]
    }

    verbs = ["make", "draw", "plot", "visualize", "show", "graph"]

    # Check for explicit chart type
    for chart, keywords in chart_keywords.items():
        for word in keywords:
            if fuzz.partial_ratio(word.lower(), query.lower()) > 80:
                return chart

    # If user uses a chart verb but no type, pick one randomly
    for verb in verbs:
        if fuzz.partial_ratio(verb.lower(), query.lower()) > 80:
            return random.choice(list(chart_keywords.keys()))

    return None

# Smart column extraction from query
def extract_smart_columns(query, dataframe, chart_type):
    columns = dataframe.columns.tolist()
    scores = [(col, fuzz.partial_ratio(col.lower(), query.lower())) for col in columns]
    sorted_cols = sorted(scores, key=lambda x: x[1], reverse=True)

    # Filter out ID-like columns and duplicates
    filtered = [col for col, score in sorted_cols if score > 60 and col.lower() not in ['id', 'index']]

    # Pie chart needs categorical + numeric
    if chart_type == "pie":
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                if pd.api.types.is_numeric_dtype(dataframe[filtered[j]]):
                    return filtered[i], filtered[j]

    # Other charts need x and y
    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered)):
            if filtered[i] != filtered[j]:
                return filtered[i], filtered[j]

    # Fallback to first two distinct columns
    fallback = [col for col in columns if col.lower() not in ['id', 'index']]
    return fallback[0], fallback[1] if len(fallback) > 1 else fallback[0]

# Generate chart using Plotly
def generate_chart(dataframe, query):
    chart_type = detect_chart_type(query)
    if chart_type:
        x, y = extract_smart_columns(query, dataframe, chart_type)

        try:
            if chart_type == "bar":
                fig = px.bar(dataframe, x=x, y=y)
            elif chart_type == "line":
                fig = px.line(dataframe, x=x, y=y)
            elif chart_type == "scatter":
                fig = px.scatter(dataframe, x=x, y=y)
            elif chart_type == "pie":
                fig = px.pie(dataframe, names=x, values=y)
            else:
                return "Chart type not supported."
            return fig
        except Exception as e:
            return f"Error generating chart: {str(e)}"
    return None

# Chat with data using PandasAI
def chat_with_data(dataframe, query):
    llm = OpenAI(api_token=openai_api_key)
    pandas_ai = PandasAI(llm)
    response = pandas_ai.run(dataframe, prompt=query)
    return response

# Streamlit UI
st.set_page_config(page_title="Total Energies", layout="wide")
st.title("💬 Total Energies - Health and Safety")

input_csv = st.file_uploader("📁 Upload your CSV file", type=['csv'])

if input_csv is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("✅ CSV Uploaded Successfully")
        data = pd.read_csv(input_csv)
        st.dataframe(data)

    with col2:
        st.info("🤖 Chat with your data")

        with st.form("chat_form", clear_on_submit=False):
            input_text = st.text_input("Enter your query")
            submitted = st.form_submit_button("Chat with your data")

        if submitted and input_text:
            st.info(f"📝 Your query: {input_text}")
            try:
                if detect_chart_type(input_text):
                    chart = generate_chart(data, input_text)
                    if isinstance(chart, str):
                        st.warning(chart)
                    else:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    result = chat_with_data(data, input_text)
                    st.success(result)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
