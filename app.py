# ============================================================
# The Data Tank - Part 2 Text Analysis Dashboard
# AI and Media Literature Analysis
# ============================================================

import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="The Data Tank - Part 2 Text Analysis Dashboard",
    layout="wide"
)

st.title("The Data Tank: AI and Media Text Analysis Dashboard")
st.markdown(
    """
    This dashboard visualises the text analysis results for Dataset 2. 
    It shows the dominant AI-media topics, sentiment patterns, stakeholder/source focus,
    and hidden sub-patterns found through topic modelling.
    """
)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
master_file = "task2_master_dashboard_data.csv"

df = pd.read_csv(master_file)

# ------------------------------------------------------------
# Clean column names / set important columns
# ------------------------------------------------------------
source_col = "Type of source (from Name (from List of sources))"

# Fill missing values
df["bert_topic_name"] = df["bert_topic_name"].fillna("Unclassified Topic")
df["Main theme"] = df["Main theme"].fillna("Unclassified Theme")
df["sentiment_label"] = df["sentiment_label"].fillna("Neutral")
df[source_col] = df[source_col].fillna("Unknown Source Type")

# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.header("Filters")

selected_topics = st.sidebar.multiselect(
    "Select AI Topics",
    options=sorted(df["bert_topic_name"].unique()),
    default=sorted(df["bert_topic_name"].unique())
)

selected_sentiments = st.sidebar.multiselect(
    "Select Sentiment",
    options=sorted(df["sentiment_label"].unique()),
    default=sorted(df["sentiment_label"].unique())
)

selected_sources = st.sidebar.multiselect(
    "Select Source Type",
    options=sorted(df[source_col].unique()),
    default=sorted(df[source_col].unique())
)

filtered_df = df[
    (df["bert_topic_name"].isin(selected_topics)) &
    (df["sentiment_label"].isin(selected_sentiments)) &
    (df[source_col].isin(selected_sources))
]

if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust the sidebar filters.")
    st.stop()

# ------------------------------------------------------------
# KPI cards
# ------------------------------------------------------------
st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Text Excerpts", len(filtered_df))

with col2:
    st.metric("AI Topics", filtered_df["bert_topic_name"].nunique())

with col3:
    st.metric("Manual Themes", filtered_df["Main theme"].nunique())

with col4:
    avg_sentiment = round(filtered_df["sentiment_score"].mean(), 3)
    st.metric("Average Sentiment", avg_sentiment)

st.markdown("---")

# ------------------------------------------------------------
# Visual 1: Dominant AI Topics
# ------------------------------------------------------------
st.subheader("1. Dominant AI and Media Topics")

topic_counts = (
    filtered_df["bert_topic_name"]
    .value_counts()
    .reset_index()
)

topic_counts.columns = ["AI Topic", "Number of Excerpts"]

fig_topic = px.bar(
    topic_counts,
    x="Number of Excerpts",
    y="AI Topic",
    orientation="h",
    text="Number of Excerpts",
    title="Most Frequent AI-Media Topics"
)

fig_topic.update_layout(
    yaxis={"categoryorder": "total ascending"},
    height=500
)

st.plotly_chart(fig_topic, use_container_width=True)

st.markdown(
    """
    This chart shows which AI and media topics appear most often in the text dataset.
    It helps identify the main focus areas in the literature, such as platform regulation,
    journalism business models, copyright/training data, and general AI-media debates.
    """
)

st.markdown("---")

# ------------------------------------------------------------
# Visual 2: Sentiment by Topic
# ------------------------------------------------------------
st.subheader("2. Sentiment by AI Topic")

topic_sentiment_filtered = (
    filtered_df
    .groupby("bert_topic_name")["sentiment_score"]
    .mean()
    .reset_index()
)

topic_sentiment_filtered.columns = ["Topic", "Avg_Sentiment"]

fig_sentiment = px.bar(
    topic_sentiment_filtered.sort_values("Avg_Sentiment"),
    x="Avg_Sentiment",
    y="Topic",
    orientation="h",
    text="Avg_Sentiment",
    title="Average Sentiment Score by AI Topic"
)

fig_sentiment.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)

fig_sentiment.update_layout(height=500)

st.plotly_chart(fig_sentiment, use_container_width=True)

st.markdown(
    """
    This chart shows whether each AI-media topic is discussed in a more positive,
    negative, or mixed way. Topics related to web scraping, crawlers, or content extraction may
    show more negative sentiment, while solution-focused topics may appear more positive.
    """
)

st.markdown("---")

# ------------------------------------------------------------
# Visual 3: Sentiment Label Breakdown
# ------------------------------------------------------------
st.subheader("3. Overall Sentiment Breakdown")

sentiment_counts = (
    filtered_df["sentiment_label"]
    .value_counts()
    .reset_index()
)

sentiment_counts.columns = ["Sentiment", "Number of Excerpts"]

fig_sentiment_pie = px.pie(
    sentiment_counts,
    names="Sentiment",
    values="Number of Excerpts",
    hole=0.4,
    title="Positive, Negative and Neutral Text Excerpts"
)

st.plotly_chart(fig_sentiment_pie, use_container_width=True)

st.markdown(
    """
    This visual summarises the overall tone of the text excerpts.
    It should not be interpreted as saying the whole AI-media debate is positive or negative.
    Instead, it shows how the coded excerpts are framed in the dataset.
    """
)

st.markdown("---")

# ------------------------------------------------------------
# Visual 4: Sentiment by Topic - Stacked Bar
# ------------------------------------------------------------
st.subheader("4. Sentiment Framing Within Each AI Topic")

topic_sentiment_stack = (
    filtered_df
    .groupby(["bert_topic_name", "sentiment_label"])
    .size()
    .reset_index(name="Count")
)

fig_stack = px.bar(
    topic_sentiment_stack,
    x="bert_topic_name",
    y="Count",
    color="sentiment_label",
    title="Sentiment Label Distribution by AI Topic",
    labels={
        "bert_topic_name": "AI Topic",
        "sentiment_label": "Sentiment",
        "Count": "Number of Excerpts"
    }
)

fig_stack.update_layout(
    xaxis_tickangle=-35,
    height=600
)

st.plotly_chart(fig_stack, use_container_width=True)

st.markdown(
    """
    This chart gives more detail than the average sentiment score.
    It shows whether a topic is mostly positive, mostly negative, or mixed.
    This is useful for explaining sensitive topics such as misinformation, copyright,
    scraping, and platform power.
    """
)

st.markdown("---")

# ------------------------------------------------------------
# Visual 5: Manual Theme Frequency
# ------------------------------------------------------------
st.subheader("5. Most Common Manual Themes")

theme_counts = (
    filtered_df["Main theme"]
    .value_counts()
    .head(15)
    .reset_index()
)

theme_counts.columns = ["Manual Theme", "Number of Excerpts"]

fig_theme = px.bar(
    theme_counts,
    x="Number of Excerpts",
    y="Manual Theme",
    orientation="h",
    text="Number of Excerpts",
    title="Top Manual Themes in the Dataset"
)

fig_theme.update_layout(
    yaxis={"categoryorder": "total ascending"},
    height=600
)

st.plotly_chart(fig_theme, use_container_width=True)

st.markdown(
    """
    This chart shows the most frequent human-coded themes.
    It helps compare the original thematic analysis with the automated AI topic modelling.
    """
)

st.markdown("---")

# ------------------------------------------------------------
# Visual 6: Manual Theme vs AI Topic Heatmap
# ------------------------------------------------------------
st.subheader("6. Manual Theme vs AI Topic Alignment")

alignment_matrix = pd.crosstab(
    filtered_df["Main theme"],
    filtered_df["bert_topic_name"],
    normalize="index"
) * 100

fig_alignment = px.imshow(
    alignment_matrix,
    text_auto=".1f",
    aspect="auto",
    title="Percentage Alignment Between Manual Themes and AI-Generated Topics",
    labels=dict(x="AI Topic", y="Manual Theme", color="Percentage")
)

fig_alignment.update_layout(height=750)

st.plotly_chart(fig_alignment, use_container_width=True)

st.markdown(
    """
    This heatmap shows how the machine-generated BERT topics overlap with the
    original manual themes. This is useful because it validates whether the NLP results are
    meaningfully connected to the existing human coding.
    """
)

st.markdown("---")

# ------------------------------------------------------------
# Visual 7: Stakeholder / Source Type vs Topic Heatmap
# ------------------------------------------------------------
st.subheader("7. Source Type vs AI Topic")

stakeholder_long = (
    filtered_df
    .groupby([source_col, "bert_topic_name"])
    .size()
    .reset_index(name="Number of Excerpts")
)

stakeholder_long.columns = [source_col, "AI Topic", "Number of Excerpts"]   

fig_stakeholder = px.density_heatmap(
    stakeholder_long,
    x="AI Topic",
    y=source_col,
    z="Number of Excerpts",
    text_auto=True,
    title="Which Source Types Discuss Which AI-Media Topics?"
)

fig_stakeholder.update_layout(
    xaxis_tickangle=-35,
    height=600
)

st.plotly_chart(fig_stakeholder, use_container_width=True)

st.markdown(
    """
    This visual connects Part 2 text analysis with the broader stakeholder/source
    perspective. It shows which source types, such as academic sources, reports, policy documents,
    or blogs/articles, focus on each AI-media topic.
    """
)

st.markdown("---")

# ------------------------------------------------------------
# Visual 8: Discovery Report
# ------------------------------------------------------------
st.subheader("8. Hidden Sub-Patterns in Manual Themes")

discovery_filtered = (
    filtered_df
    .groupby("Main theme")
    .agg(
        Unique_AI_Topics=("bert_topic_name", "nunique"),
        Number_of_Excerpts=("bert_topic_name", "count"),
        Primary_AI_Topic=("bert_topic_name", lambda x: x.value_counts().idxmax())
    )
    .reset_index()
)

discovery_filtered.columns = [
    "Manual Theme",
    "Unique_AI_Topics",
    "Number_of_Excerpts",
    "Primary_AI_Topic"
]

top_discovery = discovery_filtered.sort_values(
    "Unique_AI_Topics",
    ascending=False
).head(12)

fig_discovery = px.bar(
    top_discovery,
    x="Unique_AI_Topics",
    y="Manual Theme",
    orientation="h",
    color="Primary_AI_Topic",
    text="Unique_AI_Topics",
    title="Manual Themes Containing the Most AI-Generated Subtopics"
)

fig_discovery.update_layout(
    yaxis={"categoryorder": "total ascending"},
    height=650
)

st.plotly_chart(fig_discovery, use_container_width=True)

st.markdown(
    """
    This chart shows which manual themes are broad and contain multiple hidden
    AI-generated subtopics. It helps the client understand that some themes, such as audience loss,
    platform power, information integrity, and regulation, are complex and connected to several
    different AI-media issues.
    """
)

st.markdown("---")

# ------------------------------------------------------------
# Data table for evidence
# ------------------------------------------------------------
st.subheader("9. Evidence Table")

st.markdown(
    """
    It allows you to click through the actual
    text excerpts behind the dashboard visuals.
    """
)

show_cols = [
    "Code/Data excerpt",
    "Main theme",
    "Sub-theme",
    source_col,
    "bert_topic_name",
    "sentiment_label",
    "sentiment_score"
]

st.dataframe(
    filtered_df[show_cols],
    use_container_width=True,
    height=500
)

# ------------------------------------------------------------
# Download filtered data
# ------------------------------------------------------------
csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Dashboard Data",
    data=csv,
    file_name="filtered_task2_dashboard_data.csv",
    mime="text/csv"
)