import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import textwrap
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic  # Added the required import for BERTopic

# 1. Page Configuration
st.set_page_config(page_title="UK Energy Policy Explorer", layout="wide", page_icon="🇬🇧")
st.title("🇬🇧 UK Renewable Energy & Policy News Explorer")
st.markdown("Explore how the BBC and The Guardian cover renewable energy policy.")

# --- DICTIONARIES AND LISTS ---
# Be sure to update custom_stopwords and aspect_map with your full actual lists if needed!
COLORS = {'The Guardian': '#052962', 'BBC': '#B80000', 'Total': 'gray', 'Policy Focus': '#C27070', 'RE Focus': '#1E5631'}
custom_stopwords = ['the', 'and', 'to', 'of', 'a'] 
aspect_map = {'Policy': ['policy', 'government', 'law'], 'Renewables': ['solar', 'wind', 'green']}

# 2. Load the Data & Models
@st.cache_resource
def load_topic_model():
# 1. Get the directory where app.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Define the path to the model folder
    model_path = os.path.join(base_dir, "bertopic_model_dir")
    
    # 3. Check if the folder actually exists to avoid a crash
    if not os.path.exists(model_path):
        st.error(f"Model directory not found at: {model_path}")
        # If you are running locally in Downloads, try a fallback:
        # model_path = os.path.join(base_dir, "Downloads", "bertopic_model_dir")
        return None
        
    return BERTopic.load(model_path)

@st.cache_data
def load_articles():
    df = pd.read_csv("df_clean_with_topics.csv", encoding='utf-8-sig')
    df['published_date'] = pd.to_datetime(df['published_date'])
    return df

@st.cache_data
def load_sentiment():
    df = pd.read_csv("sentiment_lite.csv", encoding='utf-8-sig')
    df['published_date'] = pd.to_datetime(df['published_date'])
    return df

topic_model = load_topic_model()
articles_df = load_articles()
sentiment_df = load_sentiment()

# 3. Create a Sidebar for Filters
st.sidebar.header("Filter the Data")
selected_outlet = st.sidebar.multiselect(
    "Select News Outlet:", 
    options=articles_df['outlet'].unique(),
    default=articles_df['outlet'].unique()
)

filtered_articles = articles_df[articles_df['outlet'].isin(selected_outlet)]
filtered_sentiment = sentiment_df[sentiment_df['outlet'].isin(selected_outlet)]

# 4. Create Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 EDA & Coverage", 
    "🧠 Topic Modeling", 
    "😊 Sentiment Analysis", 
    "📰 Article Reader"
])

# --- TAB 1: EDA & COVERAGE ---
with tab1:
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 15 Frequent Terms")
        top_n = 15
        count_vc = CountVectorizer(ngram_range=(1, 2), stop_words=custom_stopwords, max_df=1.0)
        # Drop NaNs to prevent vectorizer errors
        clean_texts = filtered_articles['clean_text'].fillna('')
        count_matrix = count_vc.fit_transform(clean_texts)
        counts = np.asarray(count_matrix.sum(axis=0)).ravel()
        freq_df = pd.DataFrame({'term': count_vc.get_feature_names_out(), 'score': counts})
        freq_df = freq_df.sort_values(by='score', ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=freq_df, x='score', y='term', color='black', ax=ax)
        ax.set_title(f'Most {top_n} Frequent Terms', fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('')
        sns.despine()
        st.pyplot(fig, clear_figure=True)

    with col2:
        st.subheader("Top Phrases: Guardian vs BBC")
        word_stats = pd.DataFrame({
            'word': count_vc.get_feature_names_out(),
            'Guardian_Raw': count_matrix[(filtered_articles['outlet'] == 'The Guardian').to_numpy()].sum(axis=0).A1,
            'BBC_Raw': count_matrix[(filtered_articles['outlet'] == 'BBC').to_numpy()].sum(axis=0).A1
        })
        word_stats['The Guardian'] = (word_stats['Guardian_Raw'] / word_stats['Guardian_Raw'].sum()) * 1000
        word_stats['BBC'] = (word_stats['BBC_Raw'] / word_stats['BBC_Raw'].sum()) * 1000
        word_stats['ngram_type'] = word_stats['word'].apply(lambda x: 'unigram' if len(x.split()) == 1 else 'bigram')

        phrases_only = word_stats[word_stats['ngram_type'] == 'bigram']
        fig2, axes = plt.subplots(1, 2, figsize=(10, 5))
        sns.barplot(ax=axes[0], x='The Guardian', y='word', data=phrases_only.nlargest(10, 'The Guardian'), color=COLORS.get('The Guardian', 'blue'))        
        axes[0].set_title('The Guardian', fontsize=12, fontweight='bold')
        sns.barplot(ax=axes[1], x='BBC', y='word', data=phrases_only.nlargest(10, 'BBC'), color=COLORS.get('BBC', 'red'))
        axes[1].set_title('BBC', fontsize=12, fontweight='bold')
        for a in axes:
            a.set_xlabel('Freq per 1k Words')
            a.set_ylabel('')
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)

    st.divider()
    
    st.subheader("Coverage Over Time & Key Events")
    time_dist = filtered_articles.groupby([pd.Grouper(key='published_date', freq='ME'), 'outlet']).size().unstack(fill_value=0)
    if not time_dist.empty:
        time_dist['Total Coverage'] = time_dist.sum(axis=1)
        events = [
            ("Climate Emergency", "2019-05-01"), ("Net Zero Law", "2019-06-27"),
            ("10 Point Plan", "2020-11-18"), ("Energy Crisis", "2021-09-01"),
            ("Net Zero Strategy", "2021-10-19"), ("Energy Security", "2022-04-07"),
            ("GB Energy Bill", "2024-07-25"), ("Clean Power 2030", "2025-04-10")
        ] # Shortened for clarity
        
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        if 'The Guardian' in time_dist.columns:
            sns.lineplot(data=time_dist, x=time_dist.index, y='The Guardian', ax=ax3, label="The Guardian", color=COLORS.get('The Guardian', 'blue'), linewidth=2)        if 'BBC' in time_dist.columns:
        if 'BBC' in time_dist.columns:
            sns.lineplot(data=time_dist, x=time_dist.index, y='BBC', ax=ax3, label="BBC", color=COLORS.get('BBC', 'red'), linewidth=2)
        sns.lineplot(data=time_dist, x=time_dist.index, y='Total Coverage', ax=ax3, label="Total", color='gray', linestyle='--', alpha=0.5)

        for i, (label, date_str) in enumerate(events, 1):
            date_dt = pd.to_datetime(date_str)
            if not time_dist.empty and date_dt >= time_dist.index.min() and date_dt <= time_dist.index.max():
                idx = time_dist.index.get_indexer([date_dt], method='nearest')[0]
                y_val = time_dist.iloc[idx]['Total Coverage']
                ax3.plot(time_dist.index[idx], y_val, marker='o', color='black', markersize=8, zorder=3)
                ax3.text(time_dist.index[idx], y_val + (time_dist['Total Coverage'].max()*0.05), str(i), color='black', weight='bold', ha='center', fontsize=9)
        
        ax3.set_title('UK Energy Policy Online News Coverage', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Articles')
        ax3.set_xlabel('Year')
        sns.despine()
        st.pyplot(fig3, clear_figure=True)

# --- TAB 2: TOPIC MODELING ---
with tab2:
    st.header("Topic Modeling Insights")
    
    if topic_model is not None:
        st.subheader("Intertopic Distance & Hierarchy")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_dist = topic_model.visualize_topics(title="<b>Intertopic Distance Map</b>", custom_labels=True)
            st.plotly_chart(fig_dist, use_container_width=True)
        with col_t2:
            fig_hier = topic_model.visualize_hierarchy(title="<b>Topic Hierarchy</b>", custom_labels=True)
            st.plotly_chart(fig_hier, use_container_width=True)
    else:
        st.info("BERTopic model not loaded in Streamlit. Load your model to view the interactive Plotly maps.")

    st.subheader("Topics' Policy vs. Technical Density")
    topic_density = filtered_articles[filtered_articles['Topic'] != -1].groupby(['Topic', 'Topic_Label'])[['adj_policy_density', 'adj_renewable_density']].mean().reset_index().sort_values('Topic')
    
    if not topic_density.empty:
        topic_density_melted = topic_density.melt(id_vars=['Topic', 'Topic_Label'],
                                                  value_vars=['adj_policy_density', 'adj_renewable_density'],
                                                  var_name='Keyword Type', value_name='Density')
        topic_density_melted['Keyword Type'] = topic_density_melted['Keyword Type'].map({
            'adj_policy_density': 'Policy Focus', 'adj_renewable_density': 'RE Focus'})

        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=topic_density_melted, x="Density", y="Topic_Label", hue="Keyword Type", palette=COLORS, ax=ax4)
        ax4.set_xlabel("Average Term Density (per 1,000 words)")
        ax4.set_ylabel("")
        sns.despine()
        st.pyplot(fig4, clear_figure=True)

    st.subheader("Topic Focus by News Outlet")
    total_per_outlet = filtered_articles.groupby('outlet').size()
    topic_per_outlet = filtered_articles.groupby(['outlet', 'Topic', 'Topic_Label']).size().reset_index(name='count')
    if not topic_per_outlet.empty:
        topic_per_outlet['percentage'] = topic_per_outlet.apply(lambda x: (x['count'] / total_per_outlet[x['outlet']]) * 100, axis=1)
        plot_df = topic_per_outlet[topic_per_outlet['Topic'] != -1].sort_values('Topic')
        
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=plot_df, x="percentage", y="Topic_Label", hue="outlet", palette=COLORS, ax=ax5)
        ax5.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
        ax5.set_xlabel("Prevalence within Outlet's Total Coverage (%)")
        ax5.set_ylabel("")
        sns.despine()
        st.pyplot(fig5, clear_figure=True)


# --- TAB 3: SENTIMENT ANALYSIS ---
with tab3:
    st.header("Sentiment Analysis")
    sent_colors = ['#ffb3b3', '#e0e0e0', '#a5d6a7']
    order = ['Negative', 'Neutral', 'Positive']
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.subheader("Sentiment by Aspect")
        dist = filtered_sentiment.groupby(['aspect_category', 'sentiment']).size().unstack(fill_value=0)
        dist = dist.reindex(columns=order, fill_value=0)
        if not dist.empty:
            dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100
            fig6, ax6 = plt.subplots(figsize=(6, 4))
            dist_pct.plot(kind='barh', stacked=True, color=sent_colors, ax=ax6, width=0.7)
            ax6.set_xlabel('Percentage of Sentences (%)')
            ax6.set_ylabel('')
            ax6.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
            sns.despine()
            st.pyplot(fig6, clear_figure=True)

    with col_s2:
        st.subheader("Sentiment by Outlet")
        outlets = filtered_sentiment['outlet'].dropna().unique()
        fig7, axes = plt.subplots(1, len(outlets), figsize=(8, 4), sharey=True)
        if not isinstance(axes, np.ndarray): axes = [axes]
        
        for i, outlet in enumerate(outlets):
            subset = filtered_sentiment[filtered_sentiment['outlet'] == outlet]
            dist_out = subset.groupby(['aspect_category', 'sentiment']).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
            dist_out_pct = dist_out.div(dist_out.sum(axis=1), axis=0) * 100
            dist_out_pct.plot(kind='barh', stacked=True, color=sent_colors, ax=axes[i], legend=False)
            axes[i].set_title(outlet)
            axes[i].set_ylabel('')
        sns.despine()
        st.pyplot(fig7, clear_figure=True)

    st.subheader("Sentiment Trend Over Time")
    aspects = filtered_sentiment['aspect_category'].dropna().unique()
    fig8, axes8 = plt.subplots(nrows=len(aspects), ncols=1, figsize=(10, 4 * len(aspects)))
    if not isinstance(axes8, np.ndarray): axes8 = [axes8]
    
    for i, (ax, aspect) in enumerate(zip(axes8, aspects)):
        subset = filtered_sentiment[filtered_sentiment['aspect_category'] == aspect].copy().set_index('published_date')
        time_dist_s = subset.groupby([pd.Grouper(freq='QE'), 'sentiment']).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
        if not time_dist_s.empty:
            time_dist_pct = time_dist_s.div(time_dist_s.sum(axis=1), axis=0) * 100
            time_dist_pct.plot(kind='area', stacked=True, color=sent_colors, ax=ax, alpha=0.8)
            ax.set_title(f"Sentiment Trend: {aspect}", fontweight='bold')
            ax.set_ylim(0, 100)
            ax.set_ylabel('Percentage (%)')
            ax.set_xlabel('')
    plt.tight_layout()
    st.pyplot(fig8, clear_figure=True)

    st.subheader("Explore the Sentences")
    selected_sentiment = st.selectbox("Filter Sentences by Sentiment:", ["All", "Positive", "Neutral", "Negative"])
    if selected_sentiment == "All":
        st.dataframe(filtered_sentiment[['aspect_category', 'target_term', 'sentiment', 'sentence']], use_container_width=True)
    else:
        st.dataframe(filtered_sentiment[filtered_sentiment['sentiment'] == selected_sentiment][['aspect_category', 'target_term', 'sentiment', 'sentence']], use_container_width=True)


# --- TAB 4: ARTICLE READER ---
with tab4:
    st.header("Dive into the Articles")
    
    available_topics = filtered_articles['Topic_Label'].dropna().unique()
    if len(available_topics) > 0:
        selected_topic = st.selectbox("1. Filter by Topic:", available_topics)
        topic_df = filtered_articles[filtered_articles['Topic_Label'] == selected_topic]
        selected_title = st.selectbox("2. Choose an Article to Read:", topic_df['title'])
        
        if selected_title:
            article_data = topic_df[topic_df['title'] == selected_title].iloc[0]
            st.markdown(f"**Published:** {article_data['published_date'].date()} | **Outlet:** {article_data['outlet']}")
            
            with st.expander("Read Full Article Text", expanded=True):
                st.write(article_data['clean_text'])
                
            st.markdown("**Target Terms Found in this Article:**")
            article_sentiments = filtered_sentiment[filtered_sentiment['title'] == selected_title]
            if not article_sentiments.empty:
                st.dataframe(article_sentiments[['target_term', 'aspect_category', 'sentiment']], hide_index=True)
    else:
        st.info("No topic data available for the selected filters.")
