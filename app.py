import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Airline Sentiment Dashboard",
    page_icon="✈️",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("tweets_with_sentiment.csv")

df = load_data()

# ── Header ──────────────────────────────────────────────
st.title("✈️ US Airline Sentiment Analysis Dashboard")
st.markdown("Analyzing **14,000+ real tweets** to uncover public sentiment across major US airlines.")
st.divider()

# ── Sidebar filters ──────────────────────────────────────
st.sidebar.header("Filters")
airlines = st.sidebar.multiselect(
    "Select Airlines",
    options=sorted(df['airline'].unique()),
    default=sorted(df['airline'].unique())
)
sentiments = st.sidebar.multiselect(
    "Select Sentiment",
    options=["Positive", "Neutral", "Negative"],
    default=["Positive", "Neutral", "Negative"]
)

filtered_df = df[
    (df['airline'].isin(airlines)) &
    (df['predicted_sentiment'].isin(sentiments))
]

# ── KPI row ──────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tweets", f"{len(filtered_df):,}")
col2.metric("Positive", f"{len(filtered_df[filtered_df['predicted_sentiment'] == 'Positive']):,}")
col3.metric("Neutral",   f"{len(filtered_df[filtered_df['predicted_sentiment'] == 'Neutral']):,}")
col4.metric("Negative",  f"{len(filtered_df[filtered_df['predicted_sentiment'] == 'Negative']):,}")
st.divider()

# ── Charts ───────────────────────────────────────────────
color_map = {"Positive": "#2ecc71", "Neutral": "#3498db", "Negative": "#e74c3c"}

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Sentiment Distribution")
    counts = filtered_df['predicted_sentiment'].value_counts().reset_index()
    counts.columns = ['Sentiment', 'Count']
    fig1 = px.pie(
        counts, values='Count', names='Sentiment',
        color='Sentiment', color_discrete_map=color_map
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    st.subheader("Sentiment by Airline")
    airline_sent = (
        filtered_df.groupby(['airline', 'predicted_sentiment'])
        .size()
        .reset_index(name='count')
    )
    fig2 = px.bar(
        airline_sent, x='airline', y='count',
        color='predicted_sentiment',
        color_discrete_map=color_map,
        barmode='group'
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Sample tweets ────────────────────────────────────────
st.subheader("Sample Tweets")
if len(filtered_df) > 0:
    st.dataframe(
        filtered_df[['airline', 'text', 'predicted_sentiment', 'sentiment_score']]
        .sample(min(10, len(filtered_df)))
        .reset_index(drop=True),
        use_container_width=True
    )
else:
    st.warning("No tweets match your current filters.")
