import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM

st.set_page_config(layout="wide")
st.title("📈 Hidden Markov Model Stock Regime Tracker")
st.markdown("Track stock trends using Hidden Markov Models with Yahoo Finance data.")

# Sidebar Inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker (Yahoo format)", value="EKTA-B.ST")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))
n_states = st.sidebar.slider("Number of Hidden States", min_value=2, max_value=5, value=2)

# Download stock data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)
    return df

df = load_data(ticker, start_date, end_date)

# Fit HMM
model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
model.fit(df['LogReturn'].values.reshape(-1, 1))
df['State'] = model.predict(df['LogReturn'].values.reshape(-1, 1))

# Regime labeling
means = model.means_.flatten()
sorted_states = np.argsort(means)
labels = ['Bearish', 'Neutral', 'Bullish', 'Strong Bullish', 'Extreme'][:n_states]
label_map = {state: labels[i] for i, state in enumerate(sorted_states)}
df['Regime'] = df['State'].map(label_map)

# Show raw data
if st.checkbox("Show raw data"):
    st.write(df)

# Plot closing price
st.subheader("Stock Price with Regimes")
fig, ax = plt.subplots(figsize=(14, 6))
for state in df['State'].unique():
    mask = df['State'] == state
    ax.plot(df[mask].index, df[mask]['Close'], label=f"{label_map[state]} (State {state})")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Plot regime distributions
st.subheader("Log Return Distributions by Regime")
fig2, ax2 = plt.subplots(figsize=(10, 5))
for state in df['State'].unique():
    sns.kdeplot(df[df['State'] == state]['LogReturn'], label=label_map[state], ax=ax2)
ax2.set_xlabel("Log Return")
ax2.set_ylabel("Density")
ax2.legend()
st.pyplot(fig2)

# Regime Summary
st.subheader("Regime Statistics")
summary = pd.DataFrame({
    'Mean Return': model.means_.flatten(),
    'Std Dev': [np.sqrt(np.diag(cov))[0] for cov in model.covars_],
    'Frequency': [np.sum(df['State'] == i) for i in range(n_states)]
})
summary['Regime'] = [label_map[i] for i in range(n_states)]
summary = summary.sort_values(by='Mean Return', ascending=False).reset_index(drop=True)
st.dataframe(summary)

# Regime duration
st.subheader("Regime Durations")
df['Shift'] = df['State'].shift(1)
df['NewRegime'] = df['State'] != df['Shift']
df['Group'] = df['NewRegime'].cumsum()
duration_stats = df.groupby(['Group', 'State']).agg(
    Days=('Close', 'count'),
    Start=('Close', 'first'),
    End=('Close', 'last'),
    From=('Close', lambda x: x.index[0]),
    To=('Close', lambda x: x.index[-1])
).reset_index()
st.dataframe(duration_stats[['State', 'Days', 'From', 'To']])
