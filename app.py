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
def load_data():
    # Load your cleaned and processed articles
    df = pd.read_csv("df_clean_with_topics.csv", encoding='utf-8-sig')
    df['published_date'] = pd.to_datetime(df['published_date'])
    return df

df = load_data()

# 3. Create a Sidebar for Filters
st.sidebar.header("Filter the Data")
selected_outlet = st.sidebar.multiselect(
    "Select News Outlet:", 
    options=df['outlet'].unique(),
    default=df['outlet'].unique()
)

# Filter the dataframe based on user selection
filtered_df = df[df['outlet'].isin(selected_outlet)]

# 4. Show the Overall Trend
st.subheader("Coverage Over Time")
st.markdown("How much attention is renewable energy policy getting?")

# Grouping data for a simple Streamlit native line chart
time_trend = filtered_df.groupby([pd.Grouper(key='published_date', freq='ME'), 'outlet']).size().unstack(fill_value=0)
st.line_chart(time_trend)

st.divider() # Adds a horizontal line for visual separation

# 5. The Interactive Article Reader
st.subheader("📰 Dive into the Articles")
st.markdown("Select an article from the dropdown below to read the original text.")

# Let the user filter by the Topics you generated using BERTopic!
available_topics = filtered_df['Topic_Label'].unique()
selected_topic = st.selectbox("1. Filter by Topic:", available_topics)

# Narrow down articles to the selected topic
topic_df = filtered_df[filtered_df['Topic_Label'] == selected_topic]

# Let the user select a specific article title
selected_title = st.selectbox("2. Choose an Article to Read:", topic_df['title'])

# Display the selected article
if selected_title:
    # Get the specific row for this article
    article_data = topic_df[topic_df['title'] == selected_title].iloc[0]
    
    # Display Metadata
    st.markdown(f"**Published:** {article_data['published_date'].date()} | **Outlet:** {article_data['outlet']}")
    
    # Display the actual text in an expandable box to keep the UI clean
    with st.expander("Read Full Article Text", expanded=True):
        st.write(article_data['clean_text'])
