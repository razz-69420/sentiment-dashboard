# ✈️ Airline Sentiment Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-brightgreen)
![HuggingFace](https://img.shields.io/badge/HuggingFace-RoBERTa-yellow)

**Live Demo → [sentiment-dashboard.streamlit.app](https://us-airline-sentiment-analysis-dashboard.streamlit.app/)**

A sentiment analysis pipeline that processes 14,640 real tweets about US airlines using a RoBERTa transformer model fine-tuned on 124M tweets, with results visualised in an interactive filterable dashboard.

---

## Screenshots
![Dashboard Overview](screenshots/Airline Sentiment Dashboard · Streamlit.pdf)

---

## What It Does
- Preprocesses raw tweet text (replaces @mentions and URLs with model-expected tokens)
- Classifies 14,640 tweets into Positive / Neutral / Negative using HuggingFace RoBERTa
- Interactive sidebar filters by airline and sentiment
- KPI metrics, sentiment distribution pie chart, per-airline grouped bar chart, and tweet explorer

## Architecture
![Pipeline](sentiment_pipeline_architecture.pdf)

`Kaggle CSV → Pandas Preprocessing → RoBERTa Model → Labelled Dataset → Streamlit Dashboard → Streamlit Cloud`

## Tech Stack
| Layer | Tool |
|-------|------|
| Language | Python 3.13 |
| NLP Model | cardiffnlp/twitter-roberta-base-sentiment-latest |
| Dashboard | Streamlit |
| Visualisation | Plotly |
| Data | Kaggle — Twitter US Airline Sentiment (14,640 tweets) |
| Deployment | Streamlit Cloud (free) |
| Total Cost | $0.00 |

## Run Locally
```bash
git clone https://github.com/razz-69420/sentiment-dashboard
cd sentiment-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Key Findings
- 51.3% of tweets were negative, 27.2% neutral, 21.5% positive
- The model disagreed with 23.1% of the original human-assigned labels — on manual inspection, the model was more accurate in most cases, highlighting label noise common in crowdsourced datasets
- United and American Airlines had the highest proportion of negative tweets
- Delta had the most favourable sentiment ratio among all airlines

## Author
Razi Yar Khan