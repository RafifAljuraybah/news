import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(page_title="UK Energy Policy Explorer", layout="wide")
st.title("🇬🇧 UK Renewable Energy & Policy News Explorer")
st.markdown("Explore how the BBC and The Guardian cover renewable energy policy.")

# 2. Load the Data
# We use @st.cache_data so the app only loads the CSV once, making it fast!
@st.cache_data
def load_articles():
    # Load your cleaned and processed articles
    df = pd.read_csv("df_clean_with_topics.csv", encoding='utf-8-sig')
    df['published_date'] = pd.to_datetime(df['published_date'])
    return df

@st.cache_data
def load_sentiment():
    df = pd.read_csv("sentiment_lite.csv", encoding='utf-8-sig')
    df['published_date'] = pd.to_datetime(df['published_date'])
    return df
    
articles_df = load_articles()
sentiment_df = load_sentiment()

# 3. Create a Sidebar for Filters
st.sidebar.header("Filter the Data")
selected_outlet = st.sidebar.multiselect(
    "Select News Outlet:", 
    options=df['outlet'].unique(),
    default=df['outlet'].unique()
)

# Filter the dataframe based on user selection
filtered_df = df[df['outlet'].isin(selected_outlet)]
filtered_sentiment = sentiment_df[sentiment_df['outlet'].isin(selected_outlet)]

tab1, tab2, tab3 = st.tabs(["📈 Coverage & Density", "😊 Sentiment Analysis", "📰 Article Reader"])

# 4. Show the Overall Trend
with tab1:
    st.subheader("Coverage Over Time")
    time_trend = filtered_articles.groupby([pd.Grouper(key='published_date', freq='ME'), 'outlet']).size().unstack(fill_value=0)
    st.line_chart(time_trend)

    st.subheader("Average Term Density Over Time")
    st.markdown("Density of Policy vs. Renewable Energy terms (per article).")
    # Grouping density by month
    density_trend = filtered_sentiment.groupby(pd.Grouper(key='published_date', freq='ME'))[['policy_density', 'renewable_density']].mean()
    st.line_chart(density_trend)

with tab2:
    st.subheader("Overall Sentiment Distribution by Aspect")
    st.markdown("How is the tone divided between Policy terms and Renewable Energy terms?")
    
    # Create a simple sentiment distribution chart
    sentiment_counts = filtered_sentiment.groupby(['aspect_category', 'sentiment']).size().unstack(fill_value=0)
    
    # Ensure columns are in a logical order if they exist
    cols = [c for c in ['Negative', 'Neutral', 'Positive'] if c in sentiment_counts.columns]
    sentiment_counts = sentiment_counts[cols]
    
    # Plot using Streamlit's native bar chart
    st.bar_chart(sentiment_counts)

    st.subheader("Explore the Sentences")
    st.markdown("Look at the raw sentences driving these sentiments.")
    # Show a dataframe of sentences, allowing users to filter by sentiment
    selected_sentiment = st.selectbox("Filter Sentences by Sentiment:", ["All", "Positive", "Neutral", "Negative"])
    
    if selected_sentiment == "All":
        st.dataframe(filtered_sentiment[['aspect_category', 'target_term', 'sentiment', 'sentence']], use_container_width=True)
    else:
        st.dataframe(filtered_sentiment[filtered_sentiment['sentiment'] == selected_sentiment][['aspect_category', 'target_term', 'sentiment', 'sentence']], use_container_width=True)
        
# 5. The Interactive Article Reader
with tab3:
    st.subheader("Dive into the Articles")
    
    available_topics = filtered_articles['Topic_Label'].unique()
    selected_topic = st.selectbox("1. Filter by Topic:", available_topics)
    
    topic_df = filtered_articles[filtered_articles['Topic_Label'] == selected_topic]
    selected_title = st.selectbox("2. Choose an Article to Read:", topic_df['title'])
    
    if selected_title:
        article_data = topic_df[topic_df['title'] == selected_title].iloc[0]
        st.markdown(f"**Published:** {article_data['published_date'].date()} | **Outlet:** {article_data['outlet']}")
        
        with st.expander("Read Full Article Text", expanded=True):
            st.write(article_data['clean_text'])
            
        # Extra feature: Show the specific sentiment terms found in this exact article!
        st.markdown("**Target Terms Found in this Article:**")
        article_sentiments = filtered_sentiment[filtered_sentiment['title'] == selected_title]
        if not article_sentiments.empty:
            st.dataframe(article_sentiments[['target_term', 'aspect_category', 'sentiment']], hide_index=True)
