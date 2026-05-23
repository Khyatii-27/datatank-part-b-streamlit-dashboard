# ============================================================
# The Data Tank - Part B Text Analysis Dashboard
# Client-ready project outcome dashboard for Dataset 2
# ============================================================

import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="The Data Tank | AI and Media Ecosystem Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Custom styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        display: none;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    div[data-testid="stDecoration"] {
        display: none;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        background: #f4f7fb;
    }

    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0.7rem !important;
        max-width: 1420px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eef3f9 0%, #e8eef7 100%);
        border-right: 1px solid #d6e0ed;
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }

    [data-testid="stSidebar"] > div {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 0.2rem !important;
        padding-bottom: 0.7rem !important;
        margin-top: 0rem !important;
    }

    [data-testid="collapsedControl"] {
        display: none;
    }

    .top-banner {
        background: linear-gradient(135deg, #17324d 0%, #264d73 100%);
        color: white;
        border-radius: 18px;
        padding: 18px 22px;
        box-shadow: 0 10px 24px rgba(18, 40, 63, 0.16);
        margin-bottom: 12px;
    }

    .top-banner h1 {
        margin: 0 0 8px 0;
        font-size: 2.0rem;
        line-height: 1.2;
        font-weight: 700;
    }

    .top-banner p {
        margin: 0;
        color: #e7eef8;
        font-size: 1rem;
        line-height: 1.55;
    }

    .mini-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 8px;
    }

    .mini-card {
        background: #ffffff;
        border: 1px solid #d7e2ef;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 5px 14px rgba(27, 48, 71, 0.06);
    }

    .mini-card h4 {
        margin: 0 0 6px 0;
        color: #23384d;
        font-size: 1rem;
    }

    .mini-card p {
        margin: 0;
        color: #556578;
        font-size: 0.94rem;
        line-height: 1.5;
    }

    .section-box {
        background: #ffffff;
        border: 1px solid #d7e2ef;
        border-left: 4px solid #4067b1;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }

    .section-box p {
        margin: 0;
        color: #4f6174;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    .insight-box {
        background: #f8fbff;
        border: 1px solid #dbe7f6;
        border-radius: 12px;
        padding: 12px 14px;
        margin-top: 8px;
        margin-bottom: 12px;
    }

    .insight-box p {
        margin: 0;
        color: #3f536b;
        font-size: 0.94rem;
        line-height: 1.55;
    }

    .metric-intro {
        color: #66788b;
        font-size: 0.92rem;
        margin-top: -4px;
        margin-bottom: 6px;
    }

    @media (max-width: 900px) {
        .mini-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
master_file = "task2_master_dashboard_data.csv"

try:
    df = pd.read_csv(master_file)
except FileNotFoundError:
    st.error(
        "The file 'task2_master_dashboard_data.csv' was not found. Please make sure it is in the same folder as app.py."
    )
    st.stop()

# ------------------------------------------------------------
# Important columns
# ------------------------------------------------------------
source_col = "Type of source (from Name (from List of sources))"
text_col = "Code/Data excerpt"
theme_col = "Main theme"
subtheme_col = "Sub-theme"
topic_col = "bert_topic_name"
sentiment_label_col = "sentiment_label"
sentiment_score_col = "sentiment_score"

required_cols = [
    text_col,
    theme_col,
    subtheme_col,
    source_col,
    topic_col,
    sentiment_label_col,
    sentiment_score_col,
]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error("Some required columns are missing from the dataset:")
    st.write(missing_cols)
    st.stop()

# ------------------------------------------------------------
# Basic cleaning
# ------------------------------------------------------------
df[topic_col] = df[topic_col].fillna("Unclassified Topic")
df[theme_col] = df[theme_col].fillna("Unclassified Theme")
df[subtheme_col] = df[subtheme_col].fillna("Unclassified Sub-theme")
df[source_col] = df[source_col].fillna("Unknown Source Type")
df[sentiment_label_col] = df[sentiment_label_col].fillna("Neutral")
df[sentiment_score_col] = pd.to_numeric(df[sentiment_score_col], errors="coerce").fillna(0)
df[text_col] = df[text_col].fillna("No text available")

# ------------------------------------------------------------
# Plot helper
# ------------------------------------------------------------
def style_fig(fig, height=520):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fbfcfe",
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(color="#23384d"),
        title_font=dict(size=20),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e7edf6", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig

# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="margin-top:-55px; margin-bottom:18px;">
        <h2 style="margin-bottom:6px;">Dashboard Controls</h2>
        <p style="font-size:0.92rem; color:#526272; line-height:1.45; margin:0;">
        Refine the ecosystem view by topic, theme, sentiment, source type, or keyword.
        Leave a filter blank to include all values.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Small sidebar summary before filters
total_dataset_records = len(df)
st.sidebar.metric("Dataset 2 Text Excerpts", total_dataset_records)
st.sidebar.markdown("---")

with st.sidebar.expander("AI Topic Filter", expanded=True):
    selected_topics = st.multiselect(
        "Choose AI topics",
        options=sorted(df[topic_col].unique()),
        default=[],
        placeholder="All AI topics"
    )

with st.sidebar.expander("Manually Coded Theme Filter", expanded=False):
    selected_themes = st.multiselect(
        "Choose manually coded themes",
        options=sorted(df[theme_col].unique()),
        default=[],
        placeholder="All Manually Coded Themes"
    )

with st.sidebar.expander("Sentiment Filter", expanded=False):
    selected_sentiments = st.multiselect(
        "Choose sentiment labels",
        options=sorted(df[sentiment_label_col].unique()),
        default=[],
        placeholder="All sentiment labels"
    )

with st.sidebar.expander("Source Type Filter", expanded=False):
    selected_sources = st.multiselect(
        "Choose source types",
        options=sorted(df[source_col].unique()),
        default=[],
        placeholder="All source types"
    )

with st.sidebar.expander("Keyword Search", expanded=False):
    search_text = st.text_input(
        "Search text excerpts, themes, or topics",
        placeholder="Example: copyright, regulation, journalism"
    )

# Empty selection means include all values
active_topics = selected_topics if selected_topics else sorted(df[topic_col].unique())
active_themes = selected_themes if selected_themes else sorted(df[theme_col].unique())
active_sentiments = selected_sentiments if selected_sentiments else sorted(df[sentiment_label_col].unique())
active_sources = selected_sources if selected_sources else sorted(df[source_col].unique())

filtered_df = df[
    (df[topic_col].isin(active_topics))
    & (df[theme_col].isin(active_themes))
    & (df[sentiment_label_col].isin(active_sentiments))
    & (df[source_col].isin(active_sources))
].copy()

if search_text.strip():
    filtered_df = filtered_df[
        filtered_df[text_col].str.contains(search_text, case=False, na=False)
        | filtered_df[theme_col].str.contains(search_text, case=False, na=False)
        | filtered_df[subtheme_col].str.contains(search_text, case=False, na=False)
        | filtered_df[topic_col].str.contains(search_text, case=False, na=False)
    ]

# Professional filter status panel
active_filter_count = sum([
    bool(selected_topics),
    bool(selected_themes),
    bool(selected_sentiments),
    bool(selected_sources),
    bool(search_text.strip())
])

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="background:#ffffff; border:1px solid #d7e2ef; border-radius:12px; padding:12px 14px; box-shadow:0 4px 12px rgba(27,48,71,0.05);">
        <div style="font-size:0.82rem; color:#66788b; margin-bottom:4px;">Current View</div>
        <div style="font-size:1.35rem; font-weight:700; color:#1f3348;">{len(filtered_df)}</div>
        <div style="font-size:0.86rem; color:#526272;">text excerpts displayed</div>
        <div style="margin-top:8px; font-size:0.86rem; color:#526272;">Active filters: <strong>{active_filter_count}</strong></div>
    </div>
    """,
    unsafe_allow_html=True
)

if filtered_df.empty:
    st.warning("No data is available for the selected filters. Please adjust the sidebar filters.")
    st.stop()

# ------------------------------------------------------------
# Header / project outcome framing
# ------------------------------------------------------------
st.markdown(
    """
    <div class="top-banner">
        <h1>The Data Tank: AI and Media Ecosystem Dashboard</h1>
        <p>
        Visualising a dataset of recent literature excerpts on the Generative AI and media ecosystem. The dashboard helps identify dominant issues, compare manually coded themes with AI-generated topics, review sentiment framing, and trace insights back to the coded evidence
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mini-grid">
        <div class="mini-card">
            <h4>What this shows</h4>
            <p>Main issues, patterns, and relationships found in coded AI and media text data, including topics, sentiment, manually coded themes, and source perspectives.</p>
        </div>
        <div class="mini-card">
            <h4>Why it matters</h4>
            <p>It turns a coded dataset into a usable visualisation resource that can support internal analysis, stakeholder conversations, and storytelling.</p>
        </div>
        <div class="mini-card">
            <h4>Project outcome</h4>
            <p>An interactive evidence-based tool for exploring, comparing, and presenting AI and manually-coded Generative AI & media ecosystem themes in a clear and reusable format.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# KPI cards
# ------------------------------------------------------------
st.subheader("Overview")
st.markdown(
    '<div class="metric-intro">Snapshot of the currently selected subset of Dataset 2.</div>',
    unsafe_allow_html=True,
)

total_records = len(filtered_df)
unique_topics = filtered_df[topic_col].nunique()
unique_themes = filtered_df[theme_col].nunique()
unique_sources = filtered_df[source_col].nunique()
avg_sentiment = round(filtered_df[sentiment_score_col].mean(), 3)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Text Excerpts", total_records)
with c2:
    st.metric("AI Topics", unique_topics)
with c3:
    st.metric("Manually Coded Themes", unique_themes)
with c4:
    st.metric("Source Types", unique_sources)
with c5:
    st.metric("Average Sentiment", avg_sentiment)

st.markdown("---")

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Topic Overview",
        "Sentiment Analysis",
        "Manually Coded vs AI Themes",
        "Source Perspective",
        "Evidence Table",
    ]
)

# ============================================================
# TAB 1: Topic Overview
# ============================================================
with tab1:
    st.subheader("1. Dominant AI and Media Topics")
    st.markdown(
        """
        <div class="section-box">
            <p>
            This view shows which AI and media issues are most visible in the selected dataset. It helps identify the strongest areas of attention and provides a quick picture of where the conversation is concentrated.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    topic_counts = filtered_df[topic_col].value_counts().reset_index()
    topic_counts.columns = ["AI Topic", "Number of Excerpts"]
    topic_counts["Share (%)"] = (topic_counts["Number of Excerpts"] / topic_counts["Number of Excerpts"].sum() * 100).round(1)

    fig_topic = px.bar(
        topic_counts,
        x="Number of Excerpts",
        y="AI Topic",
        orientation="h",
        text="Number of Excerpts",
        title="Most Frequent AI-Media Topics",
        color="Number of Excerpts",
        color_continuous_scale="Blues",
    )
    fig_topic.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    style_fig(fig_topic, height=540)
    st.plotly_chart(fig_topic, use_container_width=True)

    top_topic = topic_counts.iloc[0]["AI Topic"]
    top_topic_count = int(topic_counts.iloc[0]["Number of Excerpts"])
    top_topic_share = topic_counts.iloc[0]["Share (%)"]

    st.markdown(
        f"""
        <div class="insight-box">
            <p><strong>Project reading:</strong> In the current selection, <strong>{top_topic}</strong> is the most prominent topic, appearing in <strong>{top_topic_count}</strong> text excerpts, or around <strong>{top_topic_share}%</strong> of the filtered data.
            This helps show where attention is most concentrated in the AI and media ecosystem and which issues may deserve deeper interpretation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_topic_pie = px.pie(
        topic_counts,
        names="AI Topic",
        values="Number of Excerpts",
        hole=0.52,
        title="Topic Composition of the Selected Data",
    )
    style_fig(fig_topic_pie, height=520)
    st.plotly_chart(fig_topic_pie, use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
            <p><strong>Why this adds value:</strong> The composition view complements the frequency chart by showing the balance between topics. This makes it easier to see whether the dataset is dominated by a few strong issues or spread more broadly across multiple parts of the AI and media ecosystem.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# TAB 2: Sentiment Analysis
# ============================================================
with tab2:
    st.subheader("2. Sentiment Analysis")
    st.markdown(
        """
        <div class="section-box">
            <p>
            This section looks at how issues are framed across the selected text excerpts. It helps show whether particular topics are discussed in more critical, neutral, or positive ways and where the tone of discussion is more mixed.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-box">
            <p>
            <strong>What sentiment means here:</strong> In this dashboard, sentiment refers to the tone of each coded text excerpt. 
            A positive sentiment suggests the excerpt frames Generative AI in a supportive or opportunity-focused way, 
            a neutral sentiment suggests a balanced or descriptive framing, and a negative sentiment suggests a more critical, 
            cautious, or risk-focused framing. The sentiment score provides a numerical indication of this tone, helping compare 
            how different AI-media topics are discussed across the dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        sentiment_counts = filtered_df[sentiment_label_col].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Number of Excerpts"]

        fig_sentiment_pie = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Number of Excerpts",
            hole=0.52,
            title="Overall Sentiment Breakdown",
        )
        style_fig(fig_sentiment_pie, height=500)
        st.plotly_chart(fig_sentiment_pie, use_container_width=True)

        dominant_sentiment = sentiment_counts.iloc[0]["Sentiment"]
        dominant_sentiment_count = int(sentiment_counts.iloc[0]["Number of Excerpts"])

        st.markdown(
            f"""
            <div class="insight-box">
                <p><strong>Project reading:</strong> <strong>{dominant_sentiment}</strong> is currently the largest sentiment category with <strong>{dominant_sentiment_count}</strong> excerpts.
                This gives a concise view of the overall tone of the selected material and helps show whether the conversation is mainly cautionary, balanced, or supportive.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        topic_sentiment_filtered = filtered_df.groupby(topic_col)[sentiment_score_col].mean().reset_index()
        topic_sentiment_filtered.columns = ["AI Topic", "Average Sentiment"]

        fig_sentiment = px.bar(
            topic_sentiment_filtered.sort_values("Average Sentiment"),
            x="Average Sentiment",
            y="AI Topic",
            orientation="h",
            text="Average Sentiment",
            title="Average Sentiment Score by AI Topic",
            color="Average Sentiment",
            color_continuous_scale="RdBu",
        )
        fig_sentiment.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_sentiment.update_layout(coloraxis_showscale=False)
        style_fig(fig_sentiment, height=500)
        st.plotly_chart(fig_sentiment, use_container_width=True)

        lowest_topic = topic_sentiment_filtered.sort_values("Average Sentiment").iloc[0]["AI Topic"]
        highest_topic = topic_sentiment_filtered.sort_values("Average Sentiment").iloc[-1]["AI Topic"]

        st.markdown(
            f"""
            <div class="insight-box">
                <p><strong>Why this matters:</strong> This comparison helps locate which topics are framed more critically and which are framed more positively.
                In the current view, <strong>{lowest_topic}</strong> sits at the lower end of sentiment, while <strong>{highest_topic}</strong> sits at the higher end. This creates a clearer way to discuss tone differences across issues.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("3. Sentiment Framing Within Each AI Topic")

    topic_sentiment_stack = filtered_df.groupby([topic_col, sentiment_label_col]).size().reset_index(name="Count")

    fig_stack = px.bar(
        topic_sentiment_stack,
        x=topic_col,
        y="Count",
        color=sentiment_label_col,
        title="Sentiment Label Distribution by AI Topic",
        labels={
            topic_col: "AI Topic",
            sentiment_label_col: "Sentiment",
            "Count": "Number of Excerpts",
        },
        barmode="stack",
    )
    fig_stack.update_layout(xaxis_tickangle=-28)
    style_fig(fig_stack, height=620)
    st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
            <p><strong>Why this is useful:</strong> A single average can hide variation. This stacked view shows whether a topic is consistently framed one way or whether it contains a mixture of positive, neutral, and negative excerpts.
            This makes it easier to communicate nuance in topics such as regulation, scraping, misinformation, platform power, and copyright.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# TAB 3: manually coded vs AI Themes
# ============================================================
with tab3:
    st.subheader("4. Most Common Manually Coded Themes")
    st.markdown(
        """
        <div class="section-box">
            <p>
            This section focuses on the manually coded thematic structure and how it connects with AI-generated topics. It helps show whether the automated topic model reinforces the manually coded analysis and where broader manually coded themes contain multiple underlying patterns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    theme_counts = filtered_df[theme_col].value_counts().head(15).reset_index()
    theme_counts.columns = ["Manually Coded Theme", "Number of Excerpts"]

    fig_theme = px.bar(
        theme_counts,
        x="Number of Excerpts",
        y="Manually Coded Theme",
        orientation="h",
        text="Number of Excerpts",
        title="Top Manually Coded Themes in the Dataset",
        color="Number of Excerpts",
        color_continuous_scale="Teal",
    )
    fig_theme.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    style_fig(fig_theme, height=620)
    st.plotly_chart(fig_theme, use_container_width=True)

    lead_theme = theme_counts.iloc[0]["Manually Coded Theme"]
    lead_theme_count = int(theme_counts.iloc[0]["Number of Excerpts"])

    st.markdown(
        f"""
        <div class="insight-box">
            <p><strong>Project reading:</strong> Manually Coded Themes represent the human-coded structure of the dataset. In the current selection, <strong>{lead_theme}</strong> is the most prominent manually coded theme with <strong>{lead_theme_count}</strong> excerpts.
            This helps show which human-coded issues are most central within the selected ecosystem view.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("5. Manually Coded Theme vs AI Topic Alignment")

    alignment_matrix = pd.crosstab(filtered_df[theme_col], filtered_df[topic_col], normalize="index") * 100

    fig_alignment = px.imshow(
        alignment_matrix,
        text_auto=".1f",
        aspect="auto",
        title="Percentage Alignment Between Manually Coded Themes and AI-Generated Topics",
        labels=dict(x="AI Topic", y="Manually Coded Theme", color="Percentage"),
    )
    style_fig(fig_alignment, height=780)
    st.plotly_chart(fig_alignment, use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
            <p><strong>Why this adds value:</strong> The heatmap shows how closely the manually coded thematic structure aligns with the AI-generated topic structure.
            Stronger cells suggest stronger overlap, while a more spread pattern suggests internal diversity. This helps explain where AI-assisted analysis supports the existing manually coded structure and where it reveals added complexity.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("6. Hidden Sub-Patterns in Manually Coded Themes")

    discovery_filtered = (
        filtered_df.groupby(theme_col)
        .agg(
            Unique_AI_Topics=(topic_col, "nunique"),
            Number_of_Excerpts=(topic_col, "count"),
            Primary_AI_Topic=(topic_col, lambda x: x.value_counts().idxmax()),
        )
        .reset_index()
    )
    discovery_filtered.columns = [
        "Manually Coded Theme",
        "Unique_AI_Topics",
        "Number_of_Excerpts",
        "Primary_AI_Topic",
    ]

    top_discovery = discovery_filtered.sort_values("Unique_AI_Topics", ascending=False).head(12)

    fig_discovery = px.bar(
        top_discovery,
        x="Unique_AI_Topics",
        y="Manually Coded Theme",
        orientation="h",
        color="Primary_AI_Topic",
        text="Unique_AI_Topics",
        title="Manually Coded Themes Containing the Most AI-Generated Subtopics",
    )
    fig_discovery.update_layout(yaxis={"categoryorder": "total ascending"})
    style_fig(fig_discovery, height=660)
    st.plotly_chart(fig_discovery, use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
            <p><strong>Why this is important:</strong> Some manually coded themes capture several distinct AI-related subtopics rather than one narrow issue.
            This chart highlights where the manually coded structure contains the richest internal variety, helping identify themes that may need deeper unpacking in future presentations, reporting, or stakeholder discussions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# TAB 4: Source Perspective
# ============================================================
with tab4:
    st.subheader("7. Source Type vs AI Topic")
    st.markdown(
        """
        <div class="section-box">
            <p>
            This section introduces a source perspective. It shows which types of sources are contributing to which AI and media discussions, helping reveal how different parts of the ecosystem shape the overall conversation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_counts = filtered_df[source_col].value_counts().reset_index()
    source_counts.columns = ["Source Type", "Number of Excerpts"]

    fig_source = px.bar(
        source_counts,
        x="Number of Excerpts",
        y="Source Type",
        orientation="h",
        text="Number of Excerpts",
        title="Text Excerpts by Source Type",
        color="Number of Excerpts",
        color_continuous_scale="Purples",
    )
    fig_source.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    style_fig(fig_source, height=540)
    st.plotly_chart(fig_source, use_container_width=True)

    lead_source = source_counts.iloc[0]["Source Type"]

    st.markdown(
        f"""
        <div class="insight-box">
            <p><strong>Project reading:</strong> This view shows which source types are most visible in the filtered dataset. In the current selection, <strong>{lead_source}</strong> is the most prominent source category.
            This matters because the overall balance of the dashboard is influenced by which source communities are contributing the most material.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stakeholder_long = filtered_df.groupby([source_col, topic_col]).size().reset_index(name="Number of Excerpts")
    stakeholder_long.columns = ["Source Type", "AI Topic", "Number of Excerpts"]

    fig_stakeholder = px.density_heatmap(
        stakeholder_long,
        x="AI Topic",
        y="Source Type",
        z="Number of Excerpts",
        text_auto=True,
        title="Which Source Types Discuss Which AI-Media Topics?",
        color_continuous_scale="Blues",
    )
    fig_stakeholder.update_layout(xaxis_tickangle=-28)
    style_fig(fig_stakeholder, height=680)
    st.plotly_chart(fig_stakeholder, use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
            <p><strong>Why this helps:</strong> The heatmap connects topic analysis with the source/stakeholder dimension of the dataset.
            It helps show whether specific issues are concentrated in certain source types or discussed more broadly across the ecosystem, creating a stronger basis for comparison and explanation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# TAB 5: Evidence Table
# ============================================================
with tab5:
    st.subheader("8. Evidence Table")
    st.markdown(
        """
        <div class="section-box">
            <p>
            This table gives direct access to the underlying coded text excerpts behind the visuals. It supports transparency, allows review of the evidence, and helps ensure that the dashboard remains grounded in the underlying dataset rather than only in summarised visuals.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    display_df = filtered_df[
        [
            text_col,
            theme_col,
            subtheme_col,
            source_col,
            topic_col,
            sentiment_label_col,
            sentiment_score_col,
        ]
    ].copy()

    st.dataframe(display_df, use_container_width=True, height=560)

    st.markdown(
        """
        <div class="insight-box">
            <p><strong>Why this table is included:</strong> The charts are designed to support pattern recognition, but interpretation should remain connected to the actual evidence.
            This table makes it possible to inspect individual excerpts, verify patterns, and use the dashboard confidently in real-world discussion or presentation settings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Dashboard Data",
        data=csv,
        file_name="filtered_task2_dashboard_data.csv",
        mime="text/csv",
    )
