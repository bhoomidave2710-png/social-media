import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# Twitter scraping
import snscrape.modules.twitter as sntwitter

# Reddit API
import praw

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Social Media Big Data Analyzer", layout="wide")

st.title("📊 Social Media Big Data Analyzer")
st.write("TF-IDF based Trending Topic Analyzer with WordCloud")

# ---------------- REDDIT AUTH ----------------
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="bigdata-analyzer"
)

# ---------------- FUNCTIONS ----------------
def tfidf_analysis(texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=100
    )
    X = vectorizer.fit_transform(texts)
    df = pd.DataFrame(
        X.toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    freq = df.sum().sort_values(ascending=False)
    return freq

def generate_wordcloud(freq):
    wc = WordCloud(
        width=900,
        height=400,
        background_color="white"
    ).generate_from_frequencies(freq.to_dict())
    return wc

# ---------------- UI TABS ----------------
tab1, tab2, tab3 = st.tabs(["🐦 Twitter", "📘 Facebook", "👽 Reddit"])

# ======================================================
# TWITTER TAB
# ======================================================
with tab1:
    st.subheader("Twitter Trending Analyzer")
    keyword = st.text_input("Enter Topic / Hashtag", "#AI")

    if st.button("Analyze Twitter"):
        tweets = []
        for i, tweet in enumerate(
            sntwitter.TwitterSearchScraper(keyword).get_items()
        ):
            if i >= 500:
                break
            tweets.append(tweet.content)

        if len(tweets) == 0:
            st.warning("No tweets found")
        else:
            df = pd.DataFrame({"Text": tweets})
            st.dataframe(df.head(20))

            freq = tfidf_analysis(tweets)

            st.subheader("TF-IDF Frequency Table")
            st.dataframe(freq.reset_index().rename(
                columns={"index": "Word", 0: "TF-IDF Score"}
            ))

            wc = generate_wordcloud(freq)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

# ======================================================
# FACEBOOK TAB (LIMITED)
# ======================================================
with tab2:
    st.subheader("Facebook Analyzer (Public Data Limitation)")
    st.info("Facebook does NOT provide free public trending APIs.")

    st.write("""
    ✔ You can integrate *Facebook Graph API*
    ✔ Requires App Review & Access Token
    ✔ Public scraping is restricted
    """)

# ======================================================
# REDDIT TAB
# ======================================================
with tab3:
    st.subheader("Reddit Trending Analyzer")
    keyword = st.text_input("Enter Topic / Keyword", "cryptocurrency")

    if st.button("Analyze Reddit"):
        posts = []
        for submission in reddit.subreddit("all").search(keyword, limit=500):
            posts.append(submission.title + " " + submission.selftext)

        if len(posts) == 0:
            st.warning("No posts found")
        else:
            df = pd.DataFrame({"Text": posts})
            st.dataframe(df.head(20))

            freq = tfidf_analysis(posts)

            st.subheader("TF-IDF Frequency Table")
            st.dataframe(freq.reset_index().rename(
                columns={"index": "Word", 0: "TF-IDF Score"}
            ))

            wc = generate_wordcloud(freq)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)
