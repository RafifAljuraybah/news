import streamlit as st
import pandas as pd
import numpy as np
import os
import textwrap
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from bertopic import BERTopic
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#1. page config
st.set_page_config(
    page_title="UK Energy Policy Explorer",
    layout="wide",
    page_icon="🇬🇧"
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
</style>
""", unsafe_allow_html=True)

st.title("UK Renewable Energy & Policy News Explorer")
st.markdown("Explore how the **BBC** and **The Guardian** cover renewable energy policy (2017–2025).")
st.metric("Total Articles Analyzed", "1,097")
st.divider()

COLORS = {
    'The Guardian': '#052962',
    'BBC': '#B80000',
    'Total Coverage': '#808080',
    'Policy Focus': '#C27070',
    'RE Focus': '#1E5631',
}
SENTIMENT_COLORS = {
    'Negative': '#ffb3b3',
    'Neutral':  '#e0e0e0',
    'Positive': '#a5d6a7',
}
events = [("Climate Emergency Declared",      "2019-05-01"),
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
("Clean Power 2030 Plan",            "2025-04-10"),]
custom_stops  = ["said", "says", "bbc", "guardian", "uk", "will"]
custom_stopwords = list(ENGLISH_STOP_WORDS.union(custom_stops))

NUM_POLICY_KEYWORDS = 72
NUM_TECH_KEYWORDS   = 89

#2. load data & models
@st.cache_resource
def load_topic_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "Downloads", "bertopic_model_dir")
    if not os.path.exists(model_path):
        st.error(f"Model directory not found at: {model_path}")
        return None
    return BERTopic.load(model_path)

@st.cache_data
def load_articles():
    df = pd.read_csv("df_clean_with_topics.csv", encoding='utf-8-sig')
    df['published_date'] = pd.to_datetime(df['published_date'])
    df['adj_policy_density']   = df['policy_density']   / NUM_POLICY_KEYWORDS
    df['adj_renewable_density'] = df['renewable_density'] / NUM_TECH_KEYWORDS
    return df

@st.cache_data
def load_sentiment():
    df = pd.read_csv("sentiment_lite.csv", encoding='utf-8-sig')
    df['published_date'] = pd.to_datetime(df['published_date'])
    return df

topic_model  = load_topic_model()
articles_df  = load_articles()
sentiment_df = load_sentiment()

#3. sidebar filters
st.sidebar.header("Filter the Data")

#3.1 outlet
all_outlets = sorted(articles_df['outlet'].dropna().unique().tolist())
selected_outlet = st.sidebar.multiselect(
    "News Outlet:",
    options=all_outlets,
    default=all_outlets,
    help="Filter articles by publisher."
)

#3.2. date range
min_date = articles_df['published_date'].min().date()
max_date = articles_df['published_date'].max().date()
date_range = st.sidebar.date_input(
    "Publication Date Range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Restrict analysis to articles published in this window."
)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = pd.Timestamp(min_date), pd.Timestamp(max_date)

#3.3 topic
all_topics = sorted(
    articles_df['Topic_Label'].dropna().unique().tolist(),
    key=lambda lbl: articles_df.loc[articles_df['Topic_Label'] == lbl, 'Topic'].min()
)
selected_topics = st.sidebar.multiselect(
    "BERTopic Topics:",
    options=all_topics,
    default=all_topics,
    help="Show only articles assigned to these topics. Topic -1 = outliers."
)

#3.4. aspect category (for sentiment tab)
all_aspects = sorted(sentiment_df['aspect_category'].dropna().unique().tolist())
selected_aspects = st.sidebar.multiselect(
    "Aspect Categories (Sentiment):",
    options=all_aspects,
    default=all_aspects,
    help="Filter sentiment charts by the aspect (Policy / Renewables / etc.)."
)

#3.5. sentiment
all_sentiments = ['Positive', 'Neutral', 'Negative']
selected_sentiments = st.sidebar.multiselect(
    "Sentiment Labels:",
    options=all_sentiments,
    default=all_sentiments,
    help="Include only these sentiment classes in the analysis."
)

st.sidebar.divider()
st.sidebar.caption(
    f"**Dataset:** {NUM_POLICY_KEYWORDS} policy keywords · "
    f"{NUM_TECH_KEYWORDS} renewable-energy keywords  \n"
    "Same keyword lists were used both to **filter** the articles "
    "and to compute **keyword densities**."
)

#apply filters
filtered_articles = articles_df[
    articles_df['outlet'].isin(selected_outlet) &
    articles_df['published_date'].between(start_date, end_date) &
    articles_df['Topic_Label'].isin(selected_topics)
].copy()

filtered_sentiment = sentiment_df[
    sentiment_df['outlet'].isin(selected_outlet) &
    sentiment_df['published_date'].between(start_date, end_date) &
    sentiment_df['aspect_category'].isin(selected_aspects) &
    sentiment_df['sentiment'].isin(selected_sentiments)
].copy()

#4. Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "EDA & Coverage",
    "Topic Modelling",
    "Sentiment Analysis",
    "Article Reader",
])

#tab 1: EDA & Coverage
with tab1:
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 15 Frequent Terms")
        top_n = 15
        count_vc    = CountVectorizer(ngram_range=(1, 2), stop_words=custom_stopwords, max_df=1.0)
        clean_texts = filtered_articles['clean_text'].fillna('')

        if len(clean_texts) > 0:
            count_matrix = count_vc.fit_transform(clean_texts)
            counts   = np.asarray(count_matrix.sum(axis=0)).ravel()
            freq_df  = pd.DataFrame({'term': count_vc.get_feature_names_out(), 'score': counts})
            freq_df  = freq_df.sort_values('score', ascending=True).tail(top_n)

            fig1 = px.bar(freq_df, x='score', y='term', orientation='h',
                          title=f'Top {top_n} Frequent Terms',
                          color_discrete_sequence=['#1a3a6b'])
            fig1.update_layout(xaxis_title='Frequency', yaxis_title='', showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No articles match the current filters.")
            count_matrix = None

    with col2:
        st.subheader("Top Phrases: Guardian vs BBC")
        if count_matrix is not None:
            guardian_mask = (filtered_articles['outlet'] == 'The Guardian').to_numpy()
            bbc_mask      = (filtered_articles['outlet'] == 'BBC').to_numpy()

            word_stats = pd.DataFrame({
                'word': count_vc.get_feature_names_out(),
                'Guardian_Raw': count_matrix[guardian_mask].sum(axis=0).A1 if guardian_mask.any() else 0,
                'BBC_Raw':      count_matrix[bbc_mask].sum(axis=0).A1      if bbc_mask.any()      else 0,
            })
            g_total = word_stats['Guardian_Raw'].sum()
            b_total = word_stats['BBC_Raw'].sum()
            word_stats['The Guardian'] = (word_stats['Guardian_Raw'] / g_total * 1000).fillna(0) if g_total else 0
            word_stats['BBC']          = (word_stats['BBC_Raw']      / b_total * 1000).fillna(0) if b_total else 0
            word_stats['ngram_type']   = word_stats['word'].apply(lambda x: 'bigram' if len(x.split()) > 1 else 'unigram')

            phrases_only = word_stats[word_stats['ngram_type'] == 'bigram']

            c1, c2 = st.columns(2)
            with c1:
                df_g = phrases_only.nlargest(10, 'The Guardian').sort_values('The Guardian', ascending=True)
                fig_g = px.bar(df_g, x='The Guardian', y='word', orientation='h',
                               title='The Guardian',
                               color_discrete_sequence=[COLORS['The Guardian']])
                fig_g.update_layout(xaxis_title='Freq per 1k Words', yaxis_title='',
                                    margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_g, use_container_width=True)

            with c2:
                df_b = phrases_only.nlargest(10, 'BBC').sort_values('BBC', ascending=True)
                fig_b = px.bar(df_b, x='BBC', y='word', orientation='h',
                               title='BBC',
                               color_discrete_sequence=[COLORS['BBC']])
                fig_b.update_layout(xaxis_title='Freq per 1k Words', yaxis_title='',
                                    margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_b, use_container_width=True)

    st.divider()
    st.subheader("Coverage Over Time & Key Policy Events")

    time_dist = (
        filtered_articles
        .groupby([pd.Grouper(key='published_date', freq='ME'), 'outlet'])
        .size()
        .unstack(fill_value=0)
    )

    if not time_dist.empty:
        time_dist['Total Coverage'] = time_dist.sum(axis=1)

        time_dist_melted = time_dist.reset_index().melt(
            id_vars='published_date', var_name='Outlet', value_name='Count')

        fig3 = px.line(
            time_dist_melted, x='published_date', y='Count', color='Outlet',
            color_discrete_map=COLORS,
            title='UK Energy Policy Online News Coverage',
        )
        for trace in fig3.data:
            if trace.name == 'Total Coverage':
                trace.line.dash  = 'dash'
                trace.line.color = 'gray'

        for i, (label, date_str) in enumerate(events, 1):
            date_dt = pd.to_datetime(date_str)
            if time_dist.index.min() <= date_dt <= time_dist.index.max():
                idx   = time_dist.index.get_indexer([date_dt], method='nearest')[0]
                y_val = time_dist.iloc[idx]['Total Coverage']
                fig3.add_annotation(
                    x=date_dt, y=y_val, text=f"<b>{i}</b>",
                    showarrow=True, arrowhead=1, ay=-40, hovertext=label
                )

        fig3.update_layout(
            xaxis_title='Year', yaxis_title='Number of Articles',
            hovermode="x unified"
        )
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("Events Key"):
            for i, (label, date_str) in enumerate(events, 1):
                st.markdown(f"**{i}.** {label} — *{date_str}*")

#tab 2 topic modelling
with tab2:
    st.header("Topic Modelling Insights")
    #  BERTopic methodology note
    with st.expander("What is BERTopic?", expanded=False):
        st.markdown("""
<div class="info-box">
<b>BERTopic</b> is an NLP topic-modelling technique that combines transformer-based sentence
embeddings (BERT) with dimensionality reduction (UMAP) and density-based clustering (HDBSCAN)
to discover coherent topics without requiring a pre-specified number of clusters.<br><br>
Each article in this corpus is assigned to exactly one topic (or to <b>Topic -1</b>,
the "outlier" bucket for articles that do not fit any coherent cluster).<br><br>
Topic labels shown here were manually assigned after inspecting the top representative
keywords and documents for each cluster.
</div>
""", unsafe_allow_html=True)

    #density methodology note
    with st.expander("How are keyword densities calculated?", expanded=False):
        st.markdown(f"""
<div class="method-box">
Two curated keyword lists were used throughout this project:

<ul>
  <li><b>Policy keywords</b> — {NUM_POLICY_KEYWORDS} terms (e.g., <i>net zero, legislation, subsidy, planning permission</i>)</li>
  <li><b>Renewable-energy keywords</b> — {NUM_TECH_KEYWORDS} terms (e.g., <i>solar, offshore wind, heat pump, battery storage</i>)</li>
</ul>

<b>Important:</b> these are the <em>same</em> lists used to filter the raw news corpus — only articles
containing at least one term from either list were included in the dataset. This means density
scores reflect relative emphasis within an already-topically-relevant sample, not the full news
output of each outlet.<br><br>

<b>Density formula</b><br>
For each article the raw keyword count is divided by the total vocabulary of its list, then
normalised to a per-1,000-word rate:

<pre>
Adjusted Density = (raw_keyword_hits / list_size) / article_word_count × 1,000
</pre>

Where <code>list_size</code> = {NUM_POLICY_KEYWORDS} for policy and {NUM_TECH_KEYWORDS} for renewables.
The bar charts below show the <b>mean</b> adjusted density across all articles in each topic cluster.
</div>
""", unsafe_allow_html=True)
    #policy vs technical density
    st.subheader("Topics: Policy vs Renewable-Energy Keyword Density")

    topic_density = (
        filtered_articles[filtered_articles['Topic'] != -1]
        .groupby(['Topic', 'Topic_Label'])[['adj_policy_density', 'adj_renewable_density']]
        .mean().reset_index().sort_values('Topic', ascending=True))

    if not topic_density.empty:
        ordered_labels = topic_density['Topic_Label'].tolist()
        topic_density_melted = topic_density.melt(
            id_vars=['Topic', 'Topic_Label'],
            value_vars=['adj_policy_density', 'adj_renewable_density'],
            var_name='Keyword Type', value_name='Density')
        topic_density_melted['Keyword Type'] = topic_density_melted['Keyword Type'].map({
            'adj_policy_density':   'Policy Focus',
            'adj_renewable_density':'RE Focus'})

        fig4 = px.bar(
            topic_density_melted, x="Density", y="Topic_Label", color="Keyword Type",
            barmode='group', orientation='h', color_discrete_map=COLORS,
            title="Mean Adjusted Keyword Density by Topic (ordered by Topic ID)")
        fig4.update_layout(
            xaxis_title="Mean Adjusted Keyword Density (hits per 1,000 words / list size)",
            yaxis_title="",
            yaxis=dict(categoryorder='array', categoryarray=ordered_labels))
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No topic data available for the selected filters.")

    #topic focus by outlet
    st.subheader("Topic Focus by News Outlet")
    total_per_outlet   = filtered_articles.groupby('outlet').size()
    topic_per_outlet   = (filtered_articles.groupby(['outlet', 'Topic', 'Topic_Label']).size().reset_index(name='count'))
    if not topic_per_outlet.empty:
        topic_per_outlet['percentage'] = topic_per_outlet.apply(
            lambda x: (x['count'] / total_per_outlet[x['outlet']]) * 100
            if x['outlet'] in total_per_outlet.index else 0,
            axis=1)
        plot_df = (topic_per_outlet[topic_per_outlet['Topic'] != -1].sort_values('Topic', ascending=True))
        ordered_topic_labels = plot_df.drop_duplicates('Topic').sort_values('Topic')['Topic_Label'].tolist()

        fig5 = px.bar(
            plot_df, x="percentage", y="Topic_Label", color="outlet",
            barmode='group', orientation='h', color_discrete_map=COLORS,
            title="Topic Prevalence within Each Outlet's Total Coverage")
        fig5.update_layout(
            xaxis_title="Share of Outlet's Total Coverage (%)",
            yaxis_title="",
            yaxis=dict(categoryorder='array', categoryarray=ordered_topic_labels))
        st.plotly_chart(fig5, use_container_width=True)

    #intertopic distance & hierarchy
    if topic_model is not None:
        st.subheader("Intertopic Distance & Hierarchy")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            try:
                fig_dist = topic_model.visualize_topics(
                    title="<b>Intertopic Distance Map</b>", custom_labels=True)
                st.plotly_chart(fig_dist, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate distance map: {e}")
        with col_t2:
            try:
                fig_hier = topic_model.visualize_hierarchy(
                    title="<b>Topic Hierarchy</b>", custom_labels=True)
                st.plotly_chart(fig_hier, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate hierarchy map: {e}")
    else:
        st.info("BERTopic model not loaded. Intertopic distance and hierarchy charts will appear once the model file is available.")


#tab 3 — Sentiment Analysis
with tab3:
    st.header("Sentiment Analysis")

    # ABSA methodology note 
    with st.expander("What is Aspect-Based Sentiment Analysis (ABSA)?", expanded=False):
        st.markdown("""
<div class="info-box">
Standard sentiment analysis assigns a single positive/negative/neutral score to an entire text.
<b>Aspect-Based Sentiment Analysis (ABSA)</b> is more granular: it identifies specific <em>aspects</em>
(topics or entities) within a sentence and scores the sentiment expressed <em>toward each aspect
independently</em>.<br><br>

In this study, two aspects were tracked across every sentence in the corpus:

<ul>
  <li><b>Policy</b> — sentences mentioning government action, legislation, planning, subsidies, etc.</li>
  <li><b>Renewables</b> — sentences mentioning solar, wind, hydrogen, heat pumps, battery storage, etc.</li>
</ul>

A sentence can carry <em>different</em> sentiments toward each aspect — for example, a sentence may
express optimism about renewable technology while being critical of government policy.<br><br>

Sentiment labels (<b>Positive / Neutral / Negative</b>) were assigned using a fine-tuned transformer
classifier run at the sentence level. The charts below aggregate these sentence-level labels
to article, outlet, and time-period granularity.
</div>
""", unsafe_allow_html=True)

    order = ['Negative', 'Neutral', 'Positive']

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.subheader("Sentiment Distribution by Aspect")
        dist = (
            filtered_sentiment
            .groupby(['aspect_category', 'sentiment'])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=order, fill_value=0))

        if not dist.empty:
            dist_pct       = dist.div(dist.sum(axis=1), axis=0) * 100
            dist_pct_reset = dist_pct.reset_index()

            fig6 = go.Figure()
            for sentiment in order:
                fig6.add_trace(go.Bar(
                    y=dist_pct_reset['aspect_category'],
                    x=dist_pct_reset[sentiment],
                    name=sentiment,
                    orientation='h',
                    marker=dict(color=SENTIMENT_COLORS[sentiment]),
                ))
            fig6.update_layout(
                barmode='stack',
                xaxis_title='% of Sentences',
                yaxis_title='',
                legend_title='Sentiment')
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("No sentiment data for current filters.")

    with col_s2:
        st.subheader("Sentiment by Outlet & Aspect")
        outlets = filtered_sentiment['outlet'].dropna().unique()

        if len(outlets) > 0:
            dist_out = (
                filtered_sentiment
                .groupby(['outlet', 'aspect_category', 'sentiment'])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=order, fill_value=0))
            dist_out_pct = dist_out.div(dist_out.sum(axis=1), axis=0) * 100
            dist_out_pct = (
                dist_out_pct
                .reset_index()
                .melt(id_vars=['outlet', 'aspect_category'],
                      value_vars=order,
                      var_name='sentiment', value_name='percentage'))

            fig7 = px.bar(
                dist_out_pct, x='percentage', y='aspect_category',
                color='sentiment', facet_col='outlet', orientation='h',
                color_discrete_map=SENTIMENT_COLORS)
            fig7.update_layout(barmode='stack', xaxis_title='% of Sentences', yaxis_title='')
            fig7.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(fig7, use_container_width=True)
        else:
            st.info("Select at least one outlet to view this chart.")

    #sentiment trend over time 
    st.subheader("Sentiment Trend Over Time (by Aspect)")
    aspects = filtered_sentiment['aspect_category'].dropna().unique()
    if len(aspects) > 0:
        for aspect in aspects:
            subset = filtered_sentiment[filtered_sentiment['aspect_category'] == aspect].copy()
            subset.set_index('published_date', inplace=True)
            time_dist_s = (
                subset
                .groupby([pd.Grouper(freq='QE'), 'sentiment'])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=order, fill_value=0))

            if not time_dist_s.empty:
                time_dist_pct = time_dist_s.div(time_dist_s.sum(axis=1), axis=0) * 100
                time_dist_pct = time_dist_pct.reset_index()

                fig8 = go.Figure()
                for sentiment in order:
                    fig8.add_trace(go.Scatter(
                        x=time_dist_pct['published_date'],
                        y=time_dist_pct[sentiment],
                        name=sentiment,
                        mode='lines',
                        stackgroup='one',
                        line=dict(width=0.5, color=SENTIMENT_COLORS[sentiment]),
                        fillcolor=SENTIMENT_COLORS[sentiment],
                    ))
                fig8.update_layout(
                    title=f"Sentiment Trend: {aspect}", xaxis_title='Quarter',
                    yaxis_title='% of Sentences', yaxis_range=[0, 100])
                st.plotly_chart(fig8, use_container_width=True)
    else:
        st.info("No data to display for the selected aspect and sentiment filters.")

    #sentence explorer 
    st.subheader("Explore Analysed Sentences")

    c_asp, c_sent, c_out2 = st.columns(3)
    with c_asp:
        exp_aspect = st.selectbox(
            "Aspect:", ["All"] + sorted(all_aspects), key="sent_explorer_aspect")
    with c_sent:
        exp_sent = st.selectbox(
            "Sentiment:", ["All"] + all_sentiments, key="sent_explorer_sent")
    with c_out2:
        exp_outlet = st.selectbox(
            "Outlet:", ["All"] + all_outlets, key="sent_explorer_outlet")

    explorer_df = filtered_sentiment.copy()
    if exp_aspect != "All":
        explorer_df = explorer_df[explorer_df['aspect_category'] == exp_aspect]
    if exp_sent != "All":
        explorer_df = explorer_df[explorer_df['sentiment'] == exp_sent]
    if exp_outlet != "All":
        explorer_df = explorer_df[explorer_df['outlet'] == exp_outlet]

    st.dataframe(
        explorer_df[['published_date', 'outlet', 'aspect_category', 'target_term', 'sentiment', 'sentence']]
        .rename(columns={
            'published_date':   'Date',
            'outlet':           'Outlet',
            'aspect_category':  'Aspect',
            'target_term':      'Target Term',
            'sentiment':        'Sentiment',
            'sentence':         'Sentence',
        })
        .reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

#tab 4
with tab4:
    st.header("📰 Article Reader")

    available_topics = (
        filtered_articles[filtered_articles['Topic'] != -1]
        ['Topic_Label'].dropna().unique().tolist()
    )
    # Sort by Topic integer
    available_topics = sorted(
        available_topics,
        key=lambda lbl: filtered_articles.loc[
            filtered_articles['Topic_Label'] == lbl, 'Topic'].min()
    )

    if len(available_topics) > 0:
        col_r1, col_r2 = st.columns([1, 2])

        with col_r1:
            selected_topic = st.selectbox("1. Filter by Topic:", available_topics)
            topic_df       = filtered_articles[filtered_articles['Topic_Label'] == selected_topic]

            outlet_filter_reader = st.selectbox(
                "2. Filter by Outlet:",
                ["All"] + sorted(topic_df['outlet'].dropna().unique().tolist()),
                key="reader_outlet",
            )
            if outlet_filter_reader != "All":
                topic_df = topic_df[topic_df['outlet'] == outlet_filter_reader]

            topic_df_sorted  = topic_df.sort_values('published_date', ascending=False)
            title_options    = topic_df_sorted['title'].dropna().tolist()
            selected_title   = st.selectbox("3. Choose an Article:", title_options)

        with col_r2:
            if selected_title:
                article_data = topic_df[topic_df['title'] == selected_title].iloc[0]

                st.markdown(f"### {article_data['title']}")
                meta_cols = st.columns(3)
                meta_cols[0].metric("Outlet",    article_data['outlet'])
                meta_cols[1].metric("Published", str(article_data['published_date'].date()))
                meta_cols[2].metric("Topic",     article_data['Topic_Label'])

                density_cols = st.columns(2)
                density_cols[0].metric(
                    "Policy Density",
                    f"{article_data['adj_policy_density']:.3f}",
                    help=f"Hits per 1k words ÷ {NUM_POLICY_KEYWORDS} policy keywords"
                )
                density_cols[1].metric(
                    "Renewables Density",
                    f"{article_data['adj_renewable_density']:.3f}",
                    help=f"Hits per 1k words ÷ {NUM_TECH_KEYWORDS} renewable keywords"
                )

                with st.expander("📄 Full Article Text", expanded=True):
                    st.write(article_data['clean_text'])

        if selected_title:
            st.divider()
            st.subheader("Aspect-Based Sentiment: Sentences in This Article")

            article_sentiments = sentiment_df[sentiment_df['title'] == selected_title].copy()

            if not article_sentiments.empty:
                # Mini sentiment bar
                counts_art = article_sentiments['sentiment'].value_counts()
                total_art  = counts_art.sum()
                bar_cols   = st.columns(3)
                for i, s in enumerate(['Positive', 'Neutral', 'Negative']):
                    n   = counts_art.get(s, 0)
                    pct = n / total_art * 100
                    bar_cols[i].metric(s, f"{n} ({pct:.0f}%)")

                # Aspect filter for this article
                aspects_in_article = sorted(article_sentiments['aspect_category'].dropna().unique().tolist())
                sel_aspect_reader  = st.selectbox(
                    "Filter by Aspect:",
                    ["All"] + aspects_in_article,
                    key="reader_aspect_filter",
                )
                if sel_aspect_reader != "All":
                    article_sentiments = article_sentiments[
                        article_sentiments['aspect_category'] == sel_aspect_reader]

                st.dataframe(
                    article_sentiments[[
                        'aspect_category', 'target_term', 'sentiment', 'sentence'
                    ]].rename(columns={
                        'aspect_category': 'Aspect',
                        'target_term':     'Target Term',
                        'sentiment':       'Sentiment',
                        'sentence':        'Sentence',
                    }).reset_index(drop=True),
                    use_container_width=True,
                    height=400,
                )
            else:
                st.info("No sentence-level sentiment data available for this article.")
    else:
        st.info("No topic data available for the selected filters.")
