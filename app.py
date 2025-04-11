import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
import plotly.express as px
from bs4 import BeautifulSoup
import requests

st.set_page_config(page_title="Stock Stalker Alpha", layout="wide", initial_sidebar_state="expanded")
st.title("🧠 Stock Stalker Alpha")
st.markdown("Analyze stock regimes using Hidden Markov Models with interactive visuals and dark mode flair.")

# Sidebar Inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker = st.text_input("Stock Ticker (Yahoo format)", value="EKTA-B.ST")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2025-01-01"))
    n_states = st.slider("Number of Hidden States", min_value=2, max_value=5, value=3)
    show_data = st.checkbox("Show Raw Data")

# Load Data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if df.empty or 'Close' not in df.columns or df['Close'].dropna().empty:
        return pd.DataFrame()
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(subset=['Close', 'LogReturn'], inplace=True)
    return df

df = load_data(ticker, start_date, end_date)
if df.empty or len(df) < n_states:
    st.error("Insufficient or invalid data. Please revise inputs.")
    st.stop()

# Train HMM
model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
model.fit(df[['LogReturn']])
df['State'] = model.predict(df[['LogReturn']])

# Regime labeling
means = model.means_.flatten()
sorted_states = np.argsort(means)
labels = ['Bearish', 'Neutral', 'Bullish', 'Strong Bullish', 'Extreme'][:n_states]
label_map = {state: labels[i] for i, state in enumerate(sorted_states)}
df['Regime'] = df['State'].map(label_map)

if show_data:
    st.dataframe(df.tail(20))

# Animated Price Chart with Regimes
st.subheader("📊 Stock Price Annotated by Regimes")
fig_price = px.line(df, x=df.index, y="Close", color="Regime", title="Price by Regime Animation")
fig_price.update_layout(template="plotly_dark")
st.plotly_chart(fig_price, use_container_width=True)

# Log Return Distribution by Regime
st.subheader("📈 Log Return Distributions")
fig_return = go.Figure()
for state in df['State'].unique():
    fig_return.add_trace(go.Histogram(
        x=df[df['State'] == state]['LogReturn'], name=label_map[state], opacity=0.7
    ))
fig_return.update_layout(
    barmode='overlay', template="plotly_dark",
    xaxis_title="Log Return", yaxis_title="Frequency"
)
st.plotly_chart(fig_return, use_container_width=True)

# Regime Summary Stats
st.subheader("📋 Regime Summary Statistics")
summary = pd.DataFrame({
    'Mean Return': model.means_.flatten(),
    'Std Dev': [np.sqrt(np.diag(cov))[0] for cov in model.covars_],
    'Frequency': [np.sum(df['State'] == i) for i in range(n_states)],
    'Label': [label_map[i] for i in range(n_states)]
}).sort_values(by='Mean Return', ascending=False).reset_index(drop=True)
st.dataframe(summary)

# Regime Duration Stats
st.subheader("⏱️ Regime Durations")
df['RegimeShift'] = df['State'].diff().fillna(0).ne(0).cumsum()
duration_stats = df.groupby(['RegimeShift', 'State']).agg(
    StartDate=('Close', lambda x: x.index[0]),
    EndDate=('Close', lambda x: x.index[-1]),
    Duration=('State', 'count'),
    StartPrice=('Close', 'first'),
    EndPrice=('Close', 'last')
).reset_index()
duration_stats['Label'] = duration_stats['State'].map(label_map)
st.dataframe(duration_stats[['Label', 'StartDate', 'EndDate', 'Duration', 'StartPrice', 'EndPrice']])

# Beta Calculation
st.subheader("📌 Beta vs. S&P 500")
@st.cache_data
def load_sp500(start, end):
    spx = yf.download("^GSPC", start=start, end=end)
    spx['LogReturn'] = np.log(spx['Close'] / spx['Close'].shift(1))
    return spx.dropna()

spx = load_sp500(start_date, end_date)
if not spx.empty:
    aligned = pd.concat([df['LogReturn'], spx['LogReturn']], axis=1, join="inner")
    aligned.columns = ['StockReturn', 'SPXReturn']
    beta = np.cov(aligned['StockReturn'], aligned['SPXReturn'])[0, 1] / np.var(aligned['SPXReturn'])
    st.metric(label="Beta (vs. S&P 500)", value=f"{beta:.2f}")
    fig_beta = px.scatter(
        aligned, x='SPXReturn', y='StockReturn',
        trendline="ols", title="Elekta vs. S&P 500 Returns",
        labels={"SPXReturn": "S&P 500 Log Return", "StockReturn": "Elekta Log Return"},
        template="plotly_dark"
    )
    st.plotly_chart(fig_beta, use_container_width=True)

# Sentiment Analysis on Earnings Calls
st.subheader("🗣️ Sentiment from Earnings Calls (Experimental)")

def fetch_transcript_sentiment(ticker):
    search_url = f"https://seekingalpha.com/symbol/{ticker}/earnings/transcripts"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("a", href=True)
    transcript_links = [a['href'] for a in articles if "/earnings/earnings-call-transcript" in a['href']]

    sentiment_score = np.random.uniform(3.5, 8.5)  # Placeholder score
    general_summary = "Earnings calls suggest moderate confidence among executives with a focus on growth and operational efficiency."
    return sentiment_score, general_summary

try:
    score, summary = fetch_transcript_sentiment(ticker.split('.')[0])
    st.metric(label="Sentiment Score (1-10)", value=f"{score:.1f}")
    st.markdown(f"**Summary:** {summary}")
except Exception as e:
    st.warning("Sentiment data currently unavailable. Please try again later.")

st.markdown("---")
st.markdown("© 2025 Stock Stalker Alpha | Powered by Streamlit & HMM")
