import feedparser
from newspaper import Article
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json, re

load_dotenv()

RSS_FEEDS = [
    "https://timesofindia.indiatimes.com/rss/topstories",
    "https://www.thehindu.com/news/national/?service=rss",
    "https://indianexpress.com/feed/",
]

def fetch_articles(max_articles=5):
    articles = []
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:2]:
            try:
                article = Article(entry.link)
                article.download()
                article.parse()
                if len(article.text) > 200:
                    articles.append({
                        "title": entry.title,
                        "text": article.text[:2000],
                        "url": entry.link
                    })
                if len(articles) >= max_articles:
                    return articles
            except:
                continue
    return articles

def generate_current_affairs(articles):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
    results = []

    for article in articles:
        prompt = f"""Based on this news article, create 2 MCQs relevant to TPSC exam preparation.

Article title: {article['title']}
Article text: {article['text']}

Return ONLY a JSON array, no explanation, no markdown:
[
  {{
    "question": "Question here?",
    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "answer": "A) option1",
    "explanation": "Brief explanation",
    "source": "{article['title']}"
  }}
]"""

        response = llm.invoke(prompt)
        text = response.content.strip()
        text = re.sub(r"```json|```", "", text).strip()
        try:
            questions = json.loads(text)
            results.extend(questions)
        except:
            continue

    return results