#  Flipkart Product Review Analyzer — Technical Report

> AI-powered sentiment & topic analysis for Flipkart product reviews. Built with Python, NLP, and Gemini AI.

[![Landing Page](screenshot_1_landing)](screenshot_1_landing)

##  Objective

Automatically extract and summarize customer sentiments, themes, and actionable insights from Flipkart product reviews — helping businesses improve products, target marketing, and understand user feedback at scale.

##  Methodology

- **Scraping**: Selenium + BeautifulSoup (anti-bot evasion)
- **NLP**: spaCy, emoji removal, grammar correction (Gemini)
- **Sentiment**: Hugging Face DistilBERT
- **Topic Modeling**: Gensim LDA
- **Insights**: Gemini AI summary
- **UI**: Streamlit

##  Live Demo

[https://jairammargam.streamlit.app](https://jairammargam.streamlit.app)

##  Report Screenshots

### 1. App Landing Page
![Landing Page](screenshot_1_landing)

### 2. Sample Raw Reviews
![Sample Reviews](screenshot_2_reviews.png)

### 3. Sentiment Distribution
![Sentiment](screenshot_3_sentiment.png)
![Sentiment Chart](screenshot_4_sentiment_chart.png)

### 4. LDA Topic Modeling
![LDA Topics](screenshot_5_lda.png)

### 5. Frequent Phrases (Bigrams/Trigrams)
![N-Grams](screenshot_6_ngrams.png)

### 6. Gemini AI Business Summary
![Gemini Insights](screenshot_7_gemini.png)

##  Key Business Takeaways

- Gain instant visibility into product issues or praise without manual review reading.
- Identify emerging patterns or complaints early through topic modeling.
- Equip product and marketing teams with actionable insights.

##  Future Enhancements

- Add sentiment-over-time tracking
- Compare multiple products/brands
- Automate email report delivery
- Add word clouds for better visual understanding

##  Conclusion

This tool empowers Flipkart and product vendors to understand the **voice of the customer at scale**, transforming unstructured reviews into **clear, actionable intelligence** using cutting-edge NLP and Generative AI.

##  GitHub Repository

[https://github.com/jairammargam/flipkart-review-analyzer](https://github.com/jairammargam/flipkart-review-analyzer)

##  Prepared By


Jai Ram Margam | Data Scientist & AI Developer
