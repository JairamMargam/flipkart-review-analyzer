# app.py
import streamlit as st
import subprocess
import sys


from review_analysis import run_full_analysis
import os

st.set_page_config(page_title="Flipkart Review Analyzer", layout="wide")
st.title("📦 Flipkart Product Review Analyzer")

st.markdown("""
Upload or provide the Flipkart **product review URL**, and our app will:
- Scrape reviews
- Clean and preprocess text
- Analyze sentiment
- Extract frequent keywords and topics
- Summarize insights using Gemini AI
""")

# --- Input section ---
product_url = st.text_input(
    "🔗 Enter Flipkart Product Review URL:",
    value="https://www.flipkart.com/nothing-phone-2a-5g-white-128-gb/product-reviews/itm85c6bca5edadc?pid=MOBGVMQSFSU7EFDH"
)

num_pages = st.slider("🔢 Number of Pages to Scrape", min_value=1, max_value=10, value=2)
use_grammar = st.checkbox("📝 Apply Grammar Correction (uses Gemini API, slower)", value=False)
api_key = st.text_input("🔑 Gemini API Key (required)", type="password")

if st.button("🚀 Analyze Reviews"):
    if not api_key:
        st.error("Please enter your Gemini API Key.")
    elif not product_url:
        st.error("Please enter a Flipkart review URL.")
    else:
        with st.spinner("Scraping and analyzing reviews. Please wait..."):
            try:
                result = run_full_analysis(
                    url=product_url,
                    num_pages=num_pages,
                    use_grammar=use_grammar,
                    api_key=api_key
                )

                st.success("✅ Analysis Complete!")

                # Show sample reviews
                st.subheader("📝 Sample Reviews")
                st.dataframe(result['raw'][['name', 'rating', 'title', 'description', 'sentiment', 'confidence']].head(10))
                
                # CSV download button
                csv = result['raw'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Reviews as CSV",
                    data=csv,
                    file_name='flipkart_reviews.csv',
                    mime='text/csv'
                )
                
                # Sentiment distribution
                st.subheader("📊 Sentiment Distribution")
                st.bar_chart(result['sentiment_counts'])

                # Topic modeling
                st.subheader("🧠 Topic Modeling (LDA)")
                st.text(result['lda_topics'])

                # Bigrams/trigrams
                st.subheader("🔍 Frequent Phrases (Bigrams/Trigrams)")
                st.text(result['bigrams_text'])

                # Gemini explanation
                st.subheader("💬 Gemini Business Summary")
                st.info(result['summary'])

                # Log for monitoring
                with open("monitoring_log.txt", "a") as f:
                    f.write(f"{st.session_state.get('run_id', 'N/A')}: Scraped {len(result['raw'])} reviews. Sentiment: {dict(result['sentiment_counts'])}\n")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)