# review_analysis.py
import pandas as pd
import numpy as np
import time
import emoji
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim import corpora
from gensim.models import LdaModel
import google.generativeai as genai
from google.generativeai import GenerativeModel
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import os
import streamlit as st

# Setup
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()
sia = SentimentIntensityAnalyzer()

def get_user_api_key(provided_key=None):
    if provided_key:
        return provided_key
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API Key is required. Please provide it via the UI or environment.")
    return api_key

import re
import requests
import json

def parse_markdown_reviews(markdown_text):
    reviews = []
    lines = [line.strip() for line in markdown_text.split('\n')]

    i = 0
    while i < len(lines):
        if re.match(r'^[1-5]\.0$', lines[i]) and (i + 4) < len(lines) and lines[i+2] == '•':
            rating = lines[i]
            title = lines[i+4]

            i += 6
            description_lines = []
            name = "Anonymous"
            location = "N/A"
            date = "N/A"

            while i < len(lines):
                if re.match(r'^[1-5]\.0$', lines[i]) and (i + 4) < len(lines) and lines[i+2] == '•':
                    break

                if lines[i].startswith("Helpful for"):
                    if len(description_lines) >= 2:
                        name_line = description_lines[-2]
                        loc_line = description_lines[-1]
                        if loc_line.startswith(", "):
                            location = loc_line[2:]
                            name = name_line
                            description_lines = description_lines[:-2]
                        else:
                            name = loc_line
                            description_lines = description_lines[:-1]
                    elif len(description_lines) >= 1:
                        name = description_lines[-1]
                        description_lines = description_lines[:-1]

                    j = i
                    while j < min(i + 5, len(lines)):
                        if lines[j].startswith("· "):
                            date = lines[j][2:]
                            break
                        j += 1

                    i = j
                    break

                if lines[i] != "":
                    description_lines.append(lines[i])
                i += 1

            desc_text = " ".join(description_lines)

            desc_text = re.sub(r'^(Review for: )?(Color|Colour).*?GB\s*', '', desc_text, flags=re.IGNORECASE).strip()
            desc_text = re.sub(r'^• Storage \d+ GB\s*', '', desc_text, flags=re.IGNORECASE).strip()
            desc_text = re.sub(r'^Review for:.*?(?=Pros|Not bad|I have|It|Excellent|Nice|Nothing|Design|It is|The|\b[A-Z])', '', desc_text, flags=re.IGNORECASE).strip()
            if desc_text.startswith("Review for:"):
                desc_text = re.sub(r'^Review for:.*?(?=\s)', '', desc_text).strip()

            desc_text = desc_text.replace("... more", "").replace("...more", "").strip()

            if desc_text:
                reviews.append({
                    'name': name,
                    'rating': rating,
                    'title': title,
                    'description': desc_text,
                    'date': date,
                    'location': location
                })
            continue
        i += 1
    return reviews


@st.cache_data(show_spinner="Scraping reviews...")
def scrape_flipkart_reviews(base_url, num_pages=2, tinyfish_api_key=None):
    if not tinyfish_api_key:
        raise ValueError("TinyFish API Key is required for scraping.")

    all_reviews = []

    urls = []
    for page in range(1, num_pages + 1):
        separator = "&" if "?" in base_url else "?"
        url = f"{base_url}{separator}page={page}"
        urls.append(url)

    headers = {
        "X-API-Key": tinyfish_api_key,
        "Content-Type": "application/json"
    }
    api_url = "https://api.fetch.tinyfish.ai"

    try:
        for i in range(0, len(urls), 10):
            batch_urls = urls[i:i+10]
            data = {
                "urls": batch_urls,
                "format": "markdown"
            }

            response = requests.post(api_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()

            result = response.json()
            if 'results' in result:
                for page in result['results']:
                    if 'text' in page and page['text']:
                        reviews = parse_markdown_reviews(page['text'])
                        all_reviews.extend(reviews)
    except Exception as e:
        print(f"Error fetching from TinyFish API: {e}")

    df = pd.DataFrame(all_reviews)
    if not df.empty:
        df = df.drop_duplicates(subset=['description']).reset_index(drop=True)
    return df

def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def clean_text(text):
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if token.is_alpha and (not token.is_stop or token.text.lower() in ['not', 'no', 'never', 'only'])])

def correct_grammar(text, api_key):
    genai.configure(api_key=api_key)
    model = GenerativeModel("gemini-1.5-flash")
    prompt = f"Correct the grammar in this sentence and return only the corrected sentence:\n\"{text}\""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Grammar correction failed: {e}")
        return text

def get_sentiment(text):
    scores = sia.polarity_scores(str(text))
    compound = scores['compound']
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    return label, abs(compound)

def run_lda(text_series, n_topics=5, label=""):
    tokenized = text_series.apply(lambda x: x.split())
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics, random_state=42, passes=10)
    topics = f"\nLDA Topics for {label} Reviews:\n"
    for idx, topic in lda.print_topics(num_words=10):
        topics += f"Topic {idx + 1}: {topic}\n"
    return topics

def get_top_ngrams(corpus, ngram_range=(2, 3), n=20):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    sorted_words = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return sorted_words[:n]

def explain_insights(bigrams, lda_topics, api_key):
    genai.configure(api_key=api_key)
    model = GenerativeModel("gemini-1.5-flash")
    prompt = f"""
I extracted review insights using these two:
Top Bigrams/Trigrams:
{bigrams}

-LDA Topics:
{lda_topics}

Now give me a *very short and clear summary* in bullet points:
What customers like
What customers dislike
What the business should improve

Keep it simple, brief, and to the point — suitable for busy users or product managers. No fluff, just insights.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Insight generation failed: {e}")
        return "Could not generate insights. Check API key or try again."

def generate_wordcloud(text_series):
    text = " ".join(text_series.dropna().astype(str))
    if not text.strip():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

@st.cache_data(show_spinner="Analyzing data...")
def run_full_analysis(url=None, num_pages=3, use_grammar=False, api_key=None, tinyfish_api_key=None):
    api_key = get_user_api_key(api_key)

    df = scrape_flipkart_reviews(url, num_pages=num_pages, tinyfish_api_key=tinyfish_api_key)

    if df.empty:
        raise ValueError("No reviews could be scraped. The page format may have changed or the URL is invalid.")
    df['description'] = df['description'].astype(str).str.strip()

    if use_grammar:
        df['description_corrected'] = df['description'].apply(lambda x: correct_grammar(x, api_key))
    else:
        df['description_corrected'] = df['description']

    df['description_no_emoji'] = df['description_corrected'].apply(remove_emojis)
    df['description_for_sentiment'] = df['description_no_emoji']
    sentiment_result = df['description_for_sentiment'].apply(get_sentiment)
    df['sentiment'] = sentiment_result.apply(lambda x: x[0])
    df['confidence'] = sentiment_result.apply(lambda x: x[1])
    df['description_cleaned'] = df['description_corrected'].apply(clean_text)

    lda_summary = run_lda(df['description_cleaned'], label="All")
    bigrams = get_top_ngrams(df['description_cleaned'])
    bigram_str = '\n'.join([f"{phrase} ({count})" for phrase, count in bigrams])
    summary = explain_insights(bigram_str, lda_summary, api_key)
    sentiment_counts = df['sentiment'].value_counts()

    return {
        'summary': summary,
        'sentiment_counts': sentiment_counts,
        'raw': df,
        'lda_topics': lda_summary,
        'bigrams_text': bigram_str
    }