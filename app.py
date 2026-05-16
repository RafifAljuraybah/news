import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from bertopic import BERTopic
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="UK Energy Policy Explorer",
    layout="wide",
    page_icon="🇬🇧",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
    .info-box {
        background: #f0f4ff;
        border-left: 4px solid #3a5fc8;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .method-box {
        background: #f5fff5;
        border-left: 4px solid #2e7d32;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .faq-box {
        background: #fffbf0;
        border-left: 4px solid #f9a825;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        line-height: 1.7;
    }
    .topic-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
    }
    .top-topic-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    .top-topic-rank { font-size: 1.3rem; }
    .top-topic-label { font-weight: 600; flex: 1; }
    .top-topic-count { color: #555; font-size: 0.88rem; white-space: nowrap; }
        .copyright-box {
        background: #fff3e0;
        border-left: 4px solid #e65100;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        font-size: 0.88rem;
        line-height: 1.7;
    }
    .disclaimer-box {
        background: #fce4ec;
        border-left: 4px solid #c62828;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-size: 0.88rem;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="copyright-box">
<b>Copyright & Data Notice</b><br>
Article titles and individual sentences displayed in this dashboard are sourced from
<b>BBC News</b> and <b>The Guardian</b> and remain the intellectual property of their
respective publishers. This content is reproduced solely for <b>non-commercial academic
research purposes</b> under fair-dealing principles.<br><br>
<b>Sources:</b><br>
• <b>The Guardian</b> , content accessed via the
  <a href="https://open-platform.theguardian.com/" target="_blank">Guardian Open Platform</a>.
  © Guardian News &amp; Media Ltd.<br>
• <b>BBC News</b> , content sourced from
  <a href="https://huggingface.co/datasets/RealTimeData/bbc_news_alltime" target="_blank">
  RealTimeData / bbc_news_alltime</a> (Hugging Face). © BBC.<br><br>
</div>
""", unsafe_allow_html=True)

COLORS = {
    'The Guardian': '#052962',
    'BBC':          '#B80000',
    'Total Coverage': '#808080',
    'Policy Focus': '#C27070',
    'RE Focus':     '#1E5631',
}
SENTIMENT_COLORS = {
    'Negative': '#ffb3b3',
    'Neutral':  '#e0e0e0',
    'Positive': '#a5d6a7',
}
NUM_POLICY_KEYWORDS = 72
NUM_TECH_KEYWORDS   = 89

EVENTS = [
    ("Climate Emergency Declared",      "2019-05-01"),
    ("UK Net Zero 2050 Law Passed",      "2019-06-27"),
    ("10 Point Plan Launch",             "2020-11-18"),
    ("Green Homes Grant Scrapped",       "2021-03-03"),
    ("Energy Crisis Starts",             "2021-09-01"),
    ("Net Zero Strategy & COP26",        "2021-10-19"),
    ("Energy Security Strategy",         "2022-04-07"),
    ("Energy Price Guarantee",           "2022-09-08"),
    ("Whitehaven Coal Mine Approved",    "2022-12-07"),
    ("Powering Up Britain Strategy",     "2023-03-30"),
    ("Net Zero Rollbacks",               "2023-09-20"),
    ("GB Energy Bill Introduction",      "2024-07-25"),
    ("COP29 & 81% Emission Target",      "2024-11-12"),
    ("Clean Power 2030 Plan",            "2025-04-10"),
]
OUTLET_HOMEPAGES = {"BBC":          "https://www.bbc.co.uk/news", "The Guardian": "https://www.theguardian.com"}
@st.cache_resource
def load_topic_model():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "Downloads", "bertopic_model_dir")
    if not os.path.exists(model_path):
        return None
    return BERTopic.load(model_path)

@st.cache_data
def load_articles():
    df = pd.read_csv("df_clean_with_topics.csv", encoding="utf-8-sig")
    df["published_date"]        = pd.to_datetime(df["published_date"])
    df["adj_policy_density"]    = df["policy_density"]   / NUM_POLICY_KEYWORDS
    df["adj_renewable_density"] = df["renewable_density"] / NUM_TECH_KEYWORDS
    return df
def _row_url(row):
    if "url" in row.index and pd.notna(row["url"]) and str(row["url"]).startswith("http"):
        return str(row["url"])
    return OUTLET_HOMEPAGES.get(row.get("outlet", ""), "#")
@st.cache_data
def load_sentiment():
    df = pd.read_csv("sentiment_lite.csv", encoding="utf-8-sig")
    df["published_date"] = pd.to_datetime(df["published_date"])
    return df

topic_model  = load_topic_model()
articles_df  = load_articles()
sentiment_df = load_sentiment()

st.title("🇬🇧 UK Renewable Energy & Policy News Explorer")
st.markdown("Explore how the **BBC** and **The Guardian** cover renewable energy policy **(2017–2025)**.")
st.divider()

# Sidebar filters
st.sidebar.header("Filter the Data")

all_outlets = sorted(articles_df["outlet"].dropna().unique().tolist())
selected_outlet = st.sidebar.multiselect(
    "News Outlet:", options=all_outlets, default=all_outlets,
    help="Filter articles by publisher.",
)

min_date = articles_df["published_date"].min().date()
max_date = articles_df["published_date"].max().date()
date_range = st.sidebar.date_input(
    "Publication Date Range:",
    value=(min_date, max_date), min_value=min_date, max_value=max_date,
)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = pd.Timestamp(min_date), pd.Timestamp(max_date)

all_aspects = sorted(sentiment_df["aspect_category"].dropna().unique().tolist())
selected_aspects = st.sidebar.multiselect(
    "Category (Sentiment):", options=all_aspects, default=all_aspects,
    help="Filter sentiment charts by topic category.",
)

all_sentiments = ["Positive", "Neutral", "Negative"]
selected_sentiments = st.sidebar.multiselect(
    "Sentiment:", options=all_sentiments, default=all_sentiments,
)

st.sidebar.divider()
st.sidebar.caption(
    f"**Dataset:** {NUM_POLICY_KEYWORDS} policy keywords · "
    f"{NUM_TECH_KEYWORDS} renewable-energy keywords"
)

# Apply filters (topics not filtered at sidebar level, use Tab 2 to explore by topic)
all_topics = sorted(
    articles_df["Topic_Label"].dropna().unique().tolist(),
    key=lambda lbl: articles_df.loc[articles_df["Topic_Label"] == lbl, "Topic"].min(),
)

filtered_articles = articles_df[
    articles_df["outlet"].isin(selected_outlet) &
    articles_df["published_date"].between(start_date, end_date) &
    articles_df["Topic_Label"].isin(all_topics)
].copy()

filtered_sentiment = sentiment_df[
    sentiment_df["outlet"].isin(selected_outlet) &
    sentiment_df["published_date"].between(start_date, end_date) &
    sentiment_df["aspect_category"].isin(selected_aspects) &
    sentiment_df["sentiment"].isin(selected_sentiments)
].copy()

tab1, tab2, tab3 = st.tabs([
    "Main Insights",
    "Topics & Articles",
    "ℹ️ About & Methods",
])

#tab 1
with tab1:

    total_filtered    = len(filtered_articles)
    guardian_filtered = len(filtered_articles[filtered_articles["outlet"] == "The Guardian"])
    bbc_filtered      = len(filtered_articles[filtered_articles["outlet"] == "BBC"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Articles", f"{total_filtered:,}")
    m2.metric("🔵 The Guardian",  f"{guardian_filtered:,}")
    m3.metric("🔴 BBC",           f"{bbc_filtered:,}")
    st.divider()

    # Coverage over time
    st.subheader("Coverage Over Time & Key Policy Events")

    time_dist = (
        filtered_articles
        .groupby([pd.Grouper(key="published_date", freq="ME"), "outlet"])
        .size()
        .unstack(fill_value=0)
    )

    if not time_dist.empty:
        time_dist["Total Coverage"] = time_dist.sum(axis=1)
        time_dist_melted = time_dist.reset_index().melt(
            id_vars="published_date", var_name="Outlet", value_name="Count"
        )

        fig_coverage = px.line(
            time_dist_melted, x="published_date", y="Count", color="Outlet",
            color_discrete_map=COLORS,
            title="UK Energy Policy Online News Coverage",
        )
        for trace in fig_coverage.data:
            if trace.name == "Total Coverage":
                trace.line.dash  = "dash"
                trace.line.color = "gray"

        for i, (label, date_str) in enumerate(EVENTS, 1):
            date_dt = pd.to_datetime(date_str)
            if time_dist.index.min() <= date_dt <= time_dist.index.max():
                idx   = time_dist.index.get_indexer([date_dt], method="nearest")[0]
                y_val = time_dist.iloc[idx]["Total Coverage"]
                fig_coverage.add_annotation(
                    x=date_dt, y=y_val, text=f"<b>{i}</b>",
                    showarrow=True, arrowhead=1, ay=-40, hovertext=label,
                )

        fig_coverage.update_layout(
            xaxis_title="Year", yaxis_title="Number of Articles",
            hovermode="x unified",
        )
        st.plotly_chart(fig_coverage, use_container_width=True)

        with st.expander("Events Key"):
            for i, (label, date_str) in enumerate(EVENTS, 1):
                st.markdown(f"**{i}.** {label} ({date_str})")
    else:
        st.info("No articles match the current filters.")

    st.divider()

    # Top 3 topics by volume , vertical layout
    st.subheader("Top 3 Topics by Volume")

    non_outlier = filtered_articles[filtered_articles["Topic"] != -1]
    if not non_outlier.empty:
        topic_counts = (
            non_outlier.groupby(["Topic", "Topic_Label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(3)
        )
        rank_labels = ["🥇", "🥈", "🥉"]
        for idx, (_, row) in enumerate(topic_counts.iterrows()):
            st.markdown(
                f"""<div class="top-topic-row">
                    <span class="top-topic-rank">{rank_labels[idx]}</span>
                    <span class="top-topic-label">{row['Topic_Label']}</span>
                    <span class="top-topic-count">{row['count']:,} articles</span>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.info("No topic data available for the selected filters.")

    st.divider()

    # Policy vs RE keyword density by topic
    st.subheader("Policy vs Renewable-Energy Keyword Density by Topic")
    st.caption(
        "Mean adjusted keyword density per topic, how heavily each topic leans "
        "on policy language vs renewable-energy technology language."
    )

    topic_density = (
        non_outlier
        .groupby(["Topic", "Topic_Label"])[["adj_policy_density", "adj_renewable_density"]]
        .mean()
        .reset_index()
        .sort_values("Topic", ascending=True)
    )

    if not topic_density.empty:
        ordered_labels = topic_density["Topic_Label"].tolist()
        topic_density_melted = topic_density.melt(
            id_vars=["Topic", "Topic_Label"],
            value_vars=["adj_policy_density", "adj_renewable_density"],
            var_name="Keyword Type", value_name="Density",
        )
        topic_density_melted["Keyword Type"] = topic_density_melted["Keyword Type"].map(
            {"adj_policy_density": "Policy Focus", "adj_renewable_density": "RE Focus"}
        )

        fig_density = px.bar(
            topic_density_melted, x="Density", y="Topic_Label", color="Keyword Type",
            barmode="group", orientation="h", color_discrete_map=COLORS,
            title="Mean Adjusted Keyword Density by Topic",
        )
        fig_density.update_layout(
            xaxis_title="Mean Adjusted Keyword Density (hits per 1,000 words / list size)",
            yaxis_title="",
            yaxis=dict(categoryorder="array", categoryarray=ordered_labels[::-1]),
        )
        st.plotly_chart(fig_density, use_container_width=True)
    else:
        st.info("No topic data for the selected filters.")

    st.divider()

    # Sentiment gap: Renewables vs Policy
    st.subheader("Sentiment Gap: Renewables vs Policy")
    st.caption(
        "Difference in % **positive** sentiment between Renewables and Policy coverage. "
        "Positive values mean renewables are framed more positively than policy; "
        "negative values mean the reverse."
    )

    if not filtered_sentiment.empty and len(all_aspects) >= 2:
        gap_df = (
            filtered_sentiment
            .groupby(["aspect_category", "sentiment"])
            .size()
            .unstack(fill_value=0)
        )
        for col in ["Positive", "Neutral", "Negative"]:
            if col not in gap_df.columns:
                gap_df[col] = 0

        gap_df["pct_positive"] = gap_df["Positive"] / gap_df.sum(axis=1) * 100
        gap_df["pct_negative"] = gap_df["Negative"] / gap_df.sum(axis=1) * 100

        aspects_present = gap_df.index.tolist()
        gap_summary = gap_df[["pct_positive", "pct_negative"]].reset_index()
        gap_summary.columns = ["Category", "% Positive", "% Negative"]
        gap_summary_melted = gap_summary.melt(
            id_vars="Category", var_name="Sentiment Type", value_name="Percentage"
        )

        color_map = {"% Positive": "#a5d6a7", "% Negative": "#ffb3b3"}
        fig_gap = px.bar(
            gap_summary_melted, x="Category", y="Percentage",
            color="Sentiment Type", barmode="group",
            color_discrete_map=color_map,
            title="% Positive vs % Negative Sentiment by Category",
        )
        fig_gap.update_layout(yaxis_title="% of Sentences", xaxis_title="")
        st.plotly_chart(fig_gap, use_container_width=True)

        if "Renewables" in aspects_present and "Policy" in aspects_present:
            re_pos  = gap_df.loc["Renewables", "pct_positive"]
            pol_pos = gap_df.loc["Policy",     "pct_positive"]
            gap_val = re_pos - pol_pos
            direction = "more positively" if gap_val > 0 else "more negatively"
            st.info(
                f"Across the filtered data, **Renewables** are framed **{abs(gap_val):.1f}% "
                f"{direction}** than **Policy** in terms of positive sentiment."
            )
    else:
        st.info("Not enough sentiment data to compute a gap. Check your filters.")

    st.divider()

    # Sentiment trend over time
    st.subheader("Sentiment Trend Over Time (Policy vs Renewables)")
    st.caption("Quarterly share of positive, neutral, and negative sentences, stacked by aspect.")

    order = ["Negative", "Neutral", "Positive"]
    aspects_in_data = filtered_sentiment["aspect_category"].dropna().unique()

    if len(aspects_in_data) > 0:
        for aspect in aspects_in_data:
            subset = filtered_sentiment[
                filtered_sentiment["aspect_category"] == aspect
            ].copy()
            subset.set_index("published_date", inplace=True)

            time_sent = (
                subset
                .groupby([pd.Grouper(freq="QE"), "sentiment"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=order, fill_value=0)
            )

            if not time_sent.empty:
                time_sent_pct = time_sent.div(time_sent.sum(axis=1), axis=0) * 100
                time_sent_pct = time_sent_pct.reset_index()

                fig_trend = go.Figure()
                for sentiment in order:
                    fig_trend.add_trace(go.Scatter(
                        x=time_sent_pct["published_date"],
                        y=time_sent_pct[sentiment],
                        name=sentiment,
                        mode="lines",
                        stackgroup="one",
                        line=dict(width=0.5, color=SENTIMENT_COLORS[sentiment]),
                        fillcolor=SENTIMENT_COLORS[sentiment],
                    ))
                fig_trend.update_layout(
                    title=f"Sentiment Trend: {aspect}",
                    xaxis_title="Quarter",
                    yaxis_title="% of Sentences",
                    yaxis_range=[0, 100],
                )
                st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No sentiment data available for the selected filters.")


#tab 2
with tab2:

    st.header("Topic Overview")
    st.caption(
        "Each topic was discovered automatically by BERTopic. The representative "
        "articles below give a flavour of what each cluster is about."
    )
    st.markdown("""
<div class="copyright-box">
<b>Content Attribution</b><br>
Article titles and sentiment sentences shown in this tab are excerpted from content
published by <b>BBC News</b> (© BBC) and <b>The Guardian</b>
(© Guardian News &amp; Media Ltd). They are reproduced here in brief for
<b>non-commercial academic research</b> purposes only. Full articles should be
accessed via the original publishers:
<a href="https://www.bbc.co.uk/news" target="_blank">bbc.co.uk/news</a> and
<a href="https://www.theguardian.com" target="_blank">theguardian.com</a>.
</div>
""", unsafe_allow_html=True)

    non_outlier_t2 = filtered_articles[filtered_articles["Topic"] != -1].copy()

    if not non_outlier_t2.empty:
        topic_summary = (
            non_outlier_t2
            .groupby(["Topic", "Topic_Label"])
            .size()
            .reset_index(name="Article Count")
            .sort_values("Topic", ascending=True)
        )

    for _, row in topic_summary.iterrows():
        topic_articles = non_outlier_t2[non_outlier_t2["Topic_Label"] == row["Topic_Label"]]
        topic_id = int(row["Topic"])
    
        # Get BERTopic's representative docs for this topic
        if topic_model is not None:
            try:
                rep_docs = topic_model.get_representative_docs(topic_id)  # list of text strings
            except Exception:
                rep_docs = []
        else:
            rep_docs = []
    
        # Match representative texts back to dataframe rows
        if rep_docs:
            rep_articles = (
                topic_articles[topic_articles["clean_text"].isin(rep_docs)]
                .drop_duplicates(subset="clean_text")
            )
            # Preserve the order BERTopic returned them in
            rep_articles = rep_articles.set_index("clean_text").reindex(
                [d for d in rep_docs if d in rep_articles["clean_text"].values]
            ).reset_index()
        else:
            rep_articles = topic_articles.sort_values("published_date", ascending=False).head(3)
        with st.expander(
            f"**{row['Topic_Label']}** · {row['Article Count']} articles", expanded=False
        ):
            if not rep_articles.empty:
                st.markdown("**Representative articles (BERTopic):**")
                for _, art in rep_articles.iterrows():
                    title   = art["title"] if pd.notna(art.get("title")) else "Untitled"
                    date    = str(art["published_date"].date()) if pd.notna(art.get("published_date")) else ""
                    outlet  = art["outlet"] if pd.notna(art.get("outlet")) else ""
                    url    = OUTLET_HOMEPAGES.get(outlet, "#")
                st.caption(
                    f"Titles © their respective publishers (BBC / The Guardian). "
                    f"Reproduced for non-commercial research purposes only.")
            else:
                st.info("No representative articles could be matched for this topic.")
                    
    else:
        st.info("No topic data available for the selected filters.")

    st.divider()
    st.header("Aspect-Based Sentiment Explorer by Category")
    st.caption(
        "Browse the individual sentences that underpin the sentiment analysis. "
        "Filter by topic, outlet, category, and sentiment label. ")

    # Copyright notice for the sentence table
    st.markdown("""
<div class="disclaimer-box">
<b>Disclaimer</b><br>
The sentences below are <b>short extracts</b> (typically one sentence) from articles
published by <b>BBC News</b> and <b>The Guardian</b>. They are displayed solely to
illustrate the sentiment classification model's output for academic purposes.
These extracts do <b>not</b> constitute a reproduction of the full journalistic work
and should not be redistributed. All intellectual property rights remain with the
original publishers.
</div>
""", unsafe_allow_html=True)

    available_topics_t2 = sorted(
        non_outlier_t2["Topic_Label"].dropna().unique().tolist(),
        key=lambda lbl: filtered_articles.loc[
            filtered_articles["Topic_Label"] == lbl, "Topic"].min())

    if len(available_topics_t2) > 0:
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)

        with col_f1:
            sel_topic_sent = st.selectbox("Filter by Topic:", ["All"] + available_topics_t2, key="sent_topic")
        with col_f2:
            sel_outlet_sent = st.selectbox(
                "Filter by Outlet:", ["All"] + sorted(filtered_articles["outlet"].dropna().unique().tolist()),
                key="sent_outlet")
        with col_f3:
            sel_aspect_sent = st.selectbox(
                "Filter by Category:",
                ["All"] + sorted(sentiment_df["aspect_category"].dropna().unique().tolist()),
                key="sent_aspect")
        with col_f4:
            sel_sentiment_sent = st.selectbox(
                "Filter by Sentiment:",
                ["All", "Positive", "Neutral", "Negative"],
                key="sent_sentiment")
        # Build filtered sentence table
        sent_display = filtered_sentiment.copy()
        if sel_topic_sent != "All":
            titles_in_topic = filtered_articles.loc[
                filtered_articles["Topic_Label"] == sel_topic_sent, "title"
            ].dropna().unique().tolist()
            sent_display = sent_display[sent_display["title"].isin(titles_in_topic)]
        if sel_outlet_sent != "All":
            sent_display = sent_display[sent_display["outlet"] == sel_outlet_sent]
        if sel_aspect_sent != "All":
            sent_display = sent_display[sent_display["aspect_category"] == sel_aspect_sent]
        if sel_sentiment_sent != "All":
            sent_display = sent_display[sent_display["sentiment"] == sel_sentiment_sent]
        st.markdown(f"**{len(sent_display):,} sentences** match the current filters.")
        if not sent_display.empty:
            # ── Article selector ──────────────────────────────────────────────
            available_titles = (
                sent_display
                .dropna(subset=["title"])
                .drop_duplicates(subset=["title"])
                .sort_values("published_date", ascending=False)["title"]
                .tolist())
            st.markdown(
                f"**{len(available_titles):,} articles** and "
                f"**{len(sent_display):,} sentences** match the current filters. "
                "Pick an article below to read its sentences.")
            sel_article = st.selectbox(
                "Select article:",
                options=available_titles,
                key="sent_article")

            article_rows = sent_display[sent_display["title"] == sel_article].copy()

            if not article_rows.empty:
                sample         = article_rows.iloc[0]
                article_url    = _row_url(sample)
                article_date   = str(sample["published_date"].date()) if pd.notna(sample.get("published_date")) else ""
                article_outlet = sample["outlet"] if pd.notna(sample.get("outlet")) else ""

                st.markdown(
                    f'📰 **[{sel_article}]({article_url}){{target="_blank"}}**  '
                    f'— {article_outlet}, {article_date}'
                )
                st.caption(f"{len(article_rows):,} sentence(s) from this article match your filters.")

                table_df = article_rows[[
                    "aspect_category", "target_term", "sentiment", "sentence",
                ]].rename(columns={
                    "aspect_category": "Category",
                    "target_term":     "Target Term",
                    "sentiment":       "Sentiment",
                    "sentence":        "Sentence",
                }).reset_index(drop=True)

                st.write(table_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                st.caption("Article titles and sentences © BBC / The Guardian.")
        else:
            st.info("No sentences match the current filters.")


#tab 3
with tab3:
    st.header("About This Dashboard")
    st.markdown(
        "This page answers the most common questions about how this dashboard works, "
        "what the numbers mean, and how to interpret the charts."
    )

    with st.expander("What is this project about?", expanded=True):
        st.markdown("""
<div class="faq-box">
This dashboard analyses <b>1,097 news articles</b> published by the <b>BBC</b> and
<b>The Guardian</b> between <b>2017 and 2025</b> that cover UK renewable energy and
climate policy.<br><br>

The goal is to understand:
<ul>
  <li>How much coverage each outlet dedicates to these topics over time</li>
  <li>What the main <em>themes</em> (topics) of coverage are</li>
  <li>Whether coverage is more focused on <em>policy & government action</em> or
      <em>renewable technology</em></li>
  <li>Whether the tone of coverage is <em>positive, neutral, or negative</em> and
      whether this differs between topics</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with st.expander("What is BERTopic and how were the topics found?"):
        st.markdown("""
<div class="info-box">
<b>BERTopic</b> is an AI-powered technique for grouping articles by theme,
think of it as a very sophisticated sorting machine.<br><br>

Here's how it works in plain English:
<ol>
  <li><b>Reading the articles</b>, Each article is converted into a list of numbers that
      capture its meaning (using a language model called BERT). Similar articles end up
      with similar numbers.</li>
  <li><b>Finding clusters</b>, Articles that are mathematically "close" to each other
      are grouped together into topics automatically. No one tells the model how many topics
      to find, it discovers this on its own.</li>
  <li><b>Labelling the topics</b>, Once clusters are found, the most representative
      keywords and articles are examined and a human-readable label is assigned to each topic.</li>
</ol>

<b>Topic -1 (Outliers)</b>, Some articles don't fit neatly into any cluster and are
placed in a catch-all "outlier" group labelled Topic -1. These are excluded from
most charts to keep things clean.
</div>
""", unsafe_allow_html=True)

    with st.expander("What does keyword density mean?"):
        st.markdown(f"""
<div class="method-box">
Keyword density measures how much an article (or a group of articles) focuses on a
particular theme, specifically, <b>policy language</b> or <b>renewable-energy technology language</b>.<br><br>

Two curated keyword lists underpin the analysis:
<ul>
  <li><b>Policy keywords</b> ({NUM_POLICY_KEYWORDS} terms), words like
      <em>net zero, legislation, subsidy, planning permission, government target</em></li>
  <li><b>Renewable-energy keywords</b> ({NUM_TECH_KEYWORDS} terms), words like
      <em>solar panels, offshore wind, heat pump, battery storage, hydrogen</em></li>
</ul>

<b>How the score is calculated:</b>
<pre>
Adjusted Density = (number of keyword hits in the article)
                   ÷ (total keywords in the list)
                   ÷ (article word count)
                   × 1,000
</pre>

Dividing by the list size makes the two scores comparable even though the lists are
different lengths. A higher score means the article uses more of that vocabulary
relative to its length.
</div>
""", unsafe_allow_html=True)

    with st.expander("What is Aspect-Based Sentiment Analysis (ABSA)?"):
        st.markdown("""
<div class="info-box">
Standard sentiment analysis gives a single score (positive / negative) to an
<em>entire</em> article. That's often too blunt.<br><br>

<b>Aspect-Based Sentiment Analysis (ABSA)</b> goes sentence by sentence and asks:
<em>"What is this sentence talking about, and what is the tone toward that specific thing?"</em><br><br>

In this project, two <b>aspects</b> are tracked in every sentence:
<ul>
  <li><b>Policy</b>, sentences about government action, legislation, subsidies,
      targets, planning rules, etc.</li>
  <li><b>Renewables</b>, sentences about solar, wind, heat pumps, hydrogen,
      battery storage, etc.</li>
</ul>

A single sentence can carry <em>different</em> tones toward each aspect. For example:<br>
<blockquote>
<em>"The government's offshore wind targets are hopelessly behind schedule,
but the technology itself continues to break cost records."</em>
</blockquote>
This sentence would be scored <b>Negative</b> toward Policy and <b>Positive</b>
toward Renewables.<br><br>

Sentiment labels were assigned by a fine-tuned AI classifier trained specifically on news text.
</div>
""", unsafe_allow_html=True)

    with st.expander("How do I read the Sentiment Gap chart?"):
        st.markdown("""
<div class="faq-box">
The Sentiment Gap chart compares the share of <b>positive</b> and <b>negative</b>
sentences for the <b>Renewables</b> aspect versus the <b>Policy</b> aspect.<br><br>

<b>What to look for:</b>
<ul>
  <li>If the <b>Renewables bar is taller</b> on the positive side, news coverage frames
      renewable technology more optimistically than government policy.</li>
  <li>If the <b>Policy bar is taller</b> on the positive side, policy is covered more
      favourably than the technology itself (less common).</li>
  <li>A large gap in the <b>negative</b> bars shows which topic receives more critical coverage.</li>
</ul>

The summary sentence below the chart tells you the gap in percentage points at a glance.
</div>
""", unsafe_allow_html=True)

    with st.expander("Where does the data come from?"):
        st.markdown("""
<div class="faq-box">
Articles were collected from two UK news outlets:

<b>The Guardian</b><br>
Articles accessed via the Guardian Open Platform API.<br>
Guardian News &amp; Media Ltd. (2017–2025). <i>UK renewable energy and climate policy
articles</i> [Dataset]. Retrieved from
<a href="https://open-platform.theguardian.com/" target="_blank">
https://open-platform.theguardian.com/</a>.<br>
© Guardian News &amp; Media Ltd. All rights reserved.<br><br>

<b>BBC News</b><br>
RealTimeData. (n.d.). <i>BBC News Alltime</i> [Dataset]. Hugging Face.
<a href="https://huggingface.co/datasets/RealTimeData/bbc_news_alltime" target="_blank">
https://huggingface.co/datasets/RealTimeData/bbc_news_alltime</a>.<br>
Original articles © BBC. All rights reserved.<br><br>

<b>Copyright statement</b><br>
This dashboard is an independent, non-commercial academic research project.
Article titles and individual sentences are reproduced in brief to illustrate
computational analysis results (topic modelling and aspect-based sentiment analysis).
Full article text is <b>not</b> reproduced anywhere in this dashboard.
Keyword density scores and sentiment labels are derived analytical outputs, not
reproductions of original content.<br><br>

<b>Date range:</b> January 2017 to June 2025<br><br>

<b>Filtering criteria:</b> Only articles containing at least <b>three terms</b> from
both the policy keyword list and the renewable-energy keyword list were included.
This ensures every article in the dataset is genuinely relevant to UK energy policy,
but it also means the dataset is a curated subset, not the complete output of either outlet.
</div>
""", unsafe_allow_html=True)
