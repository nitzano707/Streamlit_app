import streamlit as st
from transformers import pipeline

# Load the model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="avichr/heBERT_sentiment_analysis")

sentiment_analysis = load_model()

st.title("ניתוח רגשות")

text = st.text_area("הכנס טקסט כאן")

if st.button("נתח"):
    if text:
        result = sentiment_analysis(text)
        st.write(f"רגש: {result[0]['label']}")
        st.write(f"ציון: {result[0]['score']:.4f}")
    else:
        st.write("אנא הכנס טקסט לניתוח.")
