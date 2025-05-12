import streamlit as st
import requests

st.title("Tweet Sentiment Analyzer")
tweet = st.text_area("Enter tweet text", "")
if st.button("Analyze"):
    res = requests.post("http://api:8000/sentiment", json={"text": tweet})
    if res.ok:
        st.success(f"Sentiment: {res.json()['sentiment']}")
    else:
        st.error("API error")
