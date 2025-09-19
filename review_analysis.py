# review_analysis.py
import pandas as pd
import numpy as np
import time
import emoji
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
from gensim import corpora
from gensim.models import LdaModel
import google.generativeai as genai
from google.generativeai import GenerativeModel
import spacy
import nltk
import os

# Setup
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nlp = spacy.load("en_core_web_sm")

def get_user_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Gemini API Key not found in environment. Please enter it:")
        api_key = input().strip()
    if not api_key:
        raise ValueError("API Key is required.")
    return api_key

# Load sentiment model once
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def scrape_flipkart_reviews(base_url, num_pages=2):
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            delete navigator.__proto__.webdriver;
            window.navigator.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        '''
    })

    all_reviews = []

    for page in range(1, num_pages + 1):
        separator = "&" if "?" in base_url else "?"
        url = f"{base_url}{separator}page={page}"
        driver.get(url)
        time.sleep(5)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        try:
            read_more_buttons = driver.find_elements("xpath", "//span[contains(text(), 'READ MORE')]")
            for btn in read_more_buttons[:10]:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
                time.sleep(1)
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.5)
        except:
            pass

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        reviews = soup.find_all('div', class_='ZmyHeo')

        for review in reviews:
            rating_tag = review.find_previous('div', class_='XQDdHH Ga3i8K')
            rating = rating_tag.get_text(strip=True) if rating_tag else "N/A"
            title_tag = review.find_previous('p', class_='z9E0IG')
            title = title_tag.get_text(strip=True).replace("READ MORE", "").strip() if title_tag else "No Title"
            description = review.get_text(" ", strip=True).replace("READ MORE", "").strip()
            name_tag = review.find_next('p', class_='_2NsDsF AwS1CA')
            name = name_tag.get_text(strip=True) if name_tag else "Anonymous"
            location_tag = review.find_next('p', class_='MztJPv')
            location = location_tag.find_all('span')[1].get_text(strip=True) if location_tag and len(location_tag.find_all('span')) > 1 else "N/A"
            date_tag_candidates = review.find_all_next('p', class_='_2NsDsF')
            date = next((dt.get_text(strip=True) for dt in date_tag_candidates if dt.get_text(strip=True) != name), "N/A")

            all_reviews.append({
                'name': name,
                'rating': rating,
                'title': title,
                'description': description,
                'date': date,
                'location': location
            })

    driver.quit()
    df = pd.DataFrame(all_reviews)
    df = df.drop_duplicates(subset=['description']).reset_index(drop=True)
    return df

def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def clean_text(text):
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if token.is_alpha and (not token.is_stop or token.text.lower() in ['not', 'no', 'never', 'only'])])

def correct_grammar(text, api_key):
    genai.configure(api_key=api_key)
    model = GenerativeModel("gemini-2.5-flash")
    prompt = f"Correct the grammar in this sentence and return only the corrected sentence:\n\"{text}\""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Grammar correction failed: {e}")
        return text

def get_sentiment(text):
    result = sentiment_pipeline(str(text))[0]
    return result['label'], result['score']

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
    model = GenerativeModel("gemini-2.5-flash")
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

def run_full_analysis(url=None, num_pages=3, use_grammar=False, api_key=None):
    if not api_key:
        api_key = get_user_api_key()

    df = scrape_flipkart_reviews(url, num_pages=num_pages)
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