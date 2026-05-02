import streamlit as st
import pandas as pd
import numpy as np
import os
import textwrap
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic  
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Page Configuration
st.set_page_config(page_title="UK Energy Policy Explorer", layout="wide", page_icon="🇬🇧")
st.title("UK Renewable Energy & Policy News Explorer")
st.markdown("Explore how the BBC and The Guardian cover renewable energy policy.")
col1, col2, col3 = st.columns(3)
col1.metric("Total Articles Analyzed", "1,097")
col2.metric("Positive Sentiment (Renewables)", "41.9%")
col3.metric("Negative Sentiment (Policy)", "33.0%")

# --- DICTIONARIES AND LISTS ---
COLORS = {'The Guardian': '#052962', 'BBC': '#B80000', 'Total': 'gray', 'Policy Focus': '#C27070', 'RE Focus': '#1E5631'}
# Plotly uses a slightly different format for mapping, so we'll map these where appropriate.
custom_stopwords = ['the', 'and', 'to', 'of', 'a'] 
aspect_map = {'Policy': ['policy', 'government', 'law'], 'Renewables': ['solar', 'wind', 'green']}
SENTIMENT_COLORS = {'Negative': '#ffb3b3', 'Neutral': '#e0e0e0', 'Positive': '#a5d6a7'}

# 2. Load the Data & Models
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
    "EDA & Coverage", 
    "Topic Modeling", 
    "Sentiment Analysis", 
    "Article Reader"
])

# --- TAB 1: EDA & COVERAGE ---
with tab1:
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 15 Frequent Terms")
        top_n = 15
        count_vc = CountVectorizer(ngram_range=(1, 2), stop_words=custom_stopwords, max_df=1.0)
        clean_texts = filtered_articles['clean_text'].fillna('')
        count_matrix = count_vc.fit_transform(clean_texts)
        counts = np.asarray(count_matrix.sum(axis=0)).ravel()
        freq_df = pd.DataFrame({'term': count_vc.get_feature_names_out(), 'score': counts})
        freq_df = freq_df.sort_values(by='score', ascending=True).tail(top_n) # Ascending for Plotly horizontal bar

        fig1 = px.bar(freq_df, x='score', y='term', orientation='h', 
                      title=f'Most {top_n} Frequent Terms',
                      color_discrete_sequence=['#333333'])
        fig1.update_layout(xaxis_title='Frequency', yaxis_title='', showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Top Phrases: Guardian vs BBC")
        word_stats = pd.DataFrame({
            'word': count_vc.get_feature_names_out(),
            'Guardian_Raw': count_matrix[(filtered_articles['outlet'] == 'The Guardian').to_numpy()].sum(axis=0).A1 if 'The Guardian' in selected_outlet else 0,
            'BBC_Raw': count_matrix[(filtered_articles['outlet'] == 'BBC').to_numpy()].sum(axis=0).A1 if 'BBC' in selected_outlet else 0
        })
        
        word_stats['The Guardian'] = (word_stats['Guardian_Raw'] / word_stats['Guardian_Raw'].sum() * 1000).fillna(0)
        word_stats['BBC'] = (word_stats['BBC_Raw'] / word_stats['BBC_Raw'].sum() * 1000).fillna(0)
        word_stats['ngram_type'] = word_stats['word'].apply(lambda x: 'unigram' if len(x.split()) == 1 else 'bigram')

        phrases_only = word_stats[word_stats['ngram_type'] == 'bigram']
        
        # Creating two side-by-side columns within the right column for the two outlets
        c1, c2 = st.columns(2)
        with c1:
            df_g = phrases_only.nlargest(10, 'The Guardian').sort_values('The Guardian', ascending=True)
            fig_g = px.bar(df_g, x='The Guardian', y='word', orientation='h', 
                           title='The Guardian', color_discrete_sequence=[COLORS.get('The Guardian')])
            fig_g.update_layout(xaxis_title='Freq per 1k Words', yaxis_title='', margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_g, use_container_width=True)
            
        with c2:
            df_b = phrases_only.nlargest(10, 'BBC').sort_values('BBC', ascending=True)
            fig_b = px.bar(df_b, x='BBC', y='word', orientation='h', 
                           title='BBC', color_discrete_sequence=[COLORS.get('BBC')])
            fig_b.update_layout(xaxis_title='Freq per 1k Words', yaxis_title='', margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_b, use_container_width=True)

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
        ]
        
        # Melt dataframe for Plotly Express
        time_dist_melted = time_dist.reset_index().melt(id_vars='published_date', var_name='Outlet', value_name='Count')
        
        fig3 = px.line(time_dist_melted, x='published_date', y='Count', color='Outlet',
                       color_discrete_map=COLORS, title='UK Energy Policy Online News Coverage')
        
        # Update Total Coverage to be dashed
        for trace in fig3.data:
            if trace.name == 'Total Coverage':
                trace.line.dash = 'dash'
                trace.line.color = 'gray'
        
        # Add event markers
        for i, (label, date_str) in enumerate(events, 1):
            date_dt = pd.to_datetime(date_str)
            if date_dt >= time_dist.index.min() and date_dt <= time_dist.index.max():
                idx = time_dist.index.get_indexer([date_dt], method='nearest')[0]
                y_val = time_dist.iloc[idx]['Total Coverage']
                fig3.add_annotation(x=date_dt, y=y_val, text=f"<b>{i}</b>", showarrow=True, arrowhead=1, 
                                    ay=-40, hovertext=label)
                
        fig3.update_layout(xaxis_title='Year', yaxis_title='Number of Articles', hovermode="x unified")
        st.plotly_chart(fig3, use_container_width=True)

# --- TAB 2: TOPIC MODELING ---
with tab2:
    st.header("Topic Modeling Insights")
    
    if topic_model is not None:
        st.subheader("Intertopic Distance & Hierarchy")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            try:
                fig_dist = topic_model.visualize_topics(title="<b>Intertopic Distance Map</b>", custom_labels=True)
                st.plotly_chart(fig_dist, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate distance map: {e}")
        with col_t2:
            try:
                fig_hier = topic_model.visualize_hierarchy(title="<b>Topic Hierarchy</b>", custom_labels=True)
                st.plotly_chart(fig_hier, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate hierarchy map: {e}")
    else:
        st.info("BERTopic model not loaded in Streamlit. Load your model to view the interactive Plotly maps.")




    
    # --- DEBUGGING BLOCK ---
    with st.expander("🛠️ DEBUG: Why is the density chart blank?", expanded=True):
        st.write("**1. How many articles survived the Topic != -1 filter?**")
        valid_topics_df = filtered_articles[filtered_articles['Topic'] != -1]
        st.write(f"Count: {len(valid_topics_df)} articles")
        
        if len(valid_topics_df) == 0:
            st.error("Result: Your dataframe is empty because all articles are classified as Topic -1 (Outliers).")
        
        st.write("**2. Do the density columns exist in the CSV?**")
        cols_exist = 'adj_policy_density' in filtered_articles.columns and 'adj_renewable_density' in filtered_articles.columns
        st.write(f"Exist: {cols_exist}")
        
        if cols_exist:
            st.write("**3. What do the actual numbers look like? (Checking for NaNs/Zeros)**")
            st.dataframe(valid_topics_df[['Topic', 'adj_policy_density', 'adj_renewable_density']].head())
            
            st.write("**4. Checking the GroupBy and Melt step:**")
            debug_density = valid_topics_df.groupby(['Topic', 'Topic_Label'])[['adj_policy_density', 'adj_renewable_density']].mean().reset_index()
            st.dataframe(debug_density)
    # --- END DEBUGGING BLOCK ---
    st.subheader("Topics' Policy vs. Technical Density")
    topic_density = filtered_articles[filtered_articles['Topic'] != -1].groupby(['Topic', 'Topic_Label'])[['adj_policy_density', 'adj_renewable_density']].mean().reset_index().sort_values('Topic')
    
    if not topic_density.empty:
        topic_density_melted = topic_density.melt(id_vars=['Topic', 'Topic_Label'],
                                                  value_vars=['adj_policy_density', 'adj_renewable_density'],
                                                  var_name='Keyword Type', value_name='Density')
        topic_density_melted['Keyword Type'] = topic_density_melted['Keyword Type'].map({
            'adj_policy_density': 'Policy Focus', 'adj_renewable_density': 'RE Focus'})

        fig4 = px.bar(topic_density_melted, x="Density", y="Topic_Label", color="Keyword Type", 
                      barmode='group', orientation='h', color_discrete_map=COLORS)
        fig4.update_layout(xaxis_title="Average Term Density (per 1,000 words)", yaxis_title="", yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Topic Focus by News Outlet")
    total_per_outlet = filtered_articles.groupby('outlet').size()
    topic_per_outlet = filtered_articles.groupby(['outlet', 'Topic', 'Topic_Label']).size().reset_index(name='count')
    
    if not topic_per_outlet.empty:
        topic_per_outlet['percentage'] = topic_per_outlet.apply(lambda x: (x['count'] / total_per_outlet[x['outlet']]) * 100, axis=1)
        plot_df = topic_per_outlet[topic_per_outlet['Topic'] != -1].sort_values('Topic')
        
        fig5 = px.bar(plot_df, x="percentage", y="Topic_Label", color="outlet", 
                      barmode='group', orientation='h', color_discrete_map=COLORS)
        fig5.update_layout(xaxis_title="Prevalence within Outlet's Total Coverage (%)", yaxis_title="", yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig5, use_container_width=True)

# --- TAB 3: SENTIMENT ANALYSIS ---
with tab3:
    st.header("Sentiment Analysis")
    order = ['Negative', 'Neutral', 'Positive']
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.subheader("Sentiment by Aspect")
        dist = filtered_sentiment.groupby(['aspect_category', 'sentiment']).size().unstack(fill_value=0)
        dist = dist.reindex(columns=order, fill_value=0)
        
        if not dist.empty:
            dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100
            dist_pct_reset = dist_pct.reset_index()
            
            fig6 = go.Figure()
            for sentiment in order:
                fig6.add_trace(go.Bar(
                    y=dist_pct_reset['aspect_category'],
                    x=dist_pct_reset[sentiment],
                    name=sentiment,
                    orientation='h',
                    marker=dict(color=SENTIMENT_COLORS[sentiment])
                ))
            fig6.update_layout(barmode='stack', xaxis_title='Percentage of Sentences (%)', yaxis_title='')
            st.plotly_chart(fig6, use_container_width=True)

    with col_s2:
        st.subheader("Sentiment by Outlet")
        outlets = filtered_sentiment['outlet'].dropna().unique()
        
        if len(outlets) > 0:
            # Prepare data for faceting
            dist_out = filtered_sentiment.groupby(['outlet', 'aspect_category', 'sentiment']).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
            dist_out_pct = dist_out.div(dist_out.sum(axis=1), axis=0) * 100
            dist_out_pct = dist_out_pct.reset_index().melt(id_vars=['outlet', 'aspect_category'], value_vars=order, var_name='sentiment', value_name='percentage')
            
            fig7 = px.bar(dist_out_pct, x='percentage', y='aspect_category', color='sentiment', 
                          facet_col='outlet', orientation='h', 
                          color_discrete_map=SENTIMENT_COLORS)
            fig7.update_layout(barmode='stack', xaxis_title='Percentage (%)', yaxis_title='')
            fig7.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Clean facet titles
            st.plotly_chart(fig7, use_container_width=True)

    st.subheader("Sentiment Trend Over Time")
    aspects = filtered_sentiment['aspect_category'].dropna().unique()
    
    if len(aspects) > 0:
        for aspect in aspects:
            subset = filtered_sentiment[filtered_sentiment['aspect_category'] == aspect].copy()
            subset.set_index('published_date', inplace=True)
            time_dist_s = subset.groupby([pd.Grouper(freq='QE'), 'sentiment']).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
            
            if not time_dist_s.empty:
                time_dist_pct = time_dist_s.div(time_dist_s.sum(axis=1), axis=0) * 100
                time_dist_pct = time_dist_pct.reset_index()
                
                fig8 = go.Figure()
                for sentiment in order:
                    fig8.add_trace(go.Scatter(
                        x=time_dist_pct['published_date'], y=time_dist_pct[sentiment],
                        name=sentiment,
                        mode='lines',
                        stackgroup='one',
                        line=dict(width=0.5, color=SENTIMENT_COLORS[sentiment]),
                        fillcolor=SENTIMENT_COLORS[sentiment]
                    ))
                fig8.update_layout(title=f"Sentiment Trend: {aspect}", xaxis_title='', yaxis_title='Percentage (%)', yaxis_range=[0, 100])
                st.plotly_chart(fig8, use_container_width=True)

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
