import logging
import certifi
import requests
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from deep_translator import GoogleTranslator  # Synchronous translator

class GDELTNewsClient:
    def __init__(self, base_url="https://api.gdeltproject.org/api/v2/doc/doc"):
        self.base_url = base_url
        self.cache = {}  # Cache headlines keyed by date string (YYYYMMDD)
        self.headers = {"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"}

    def fetch_headlines(self, query="bitcoin", page_size=5, date=None):
        cache_key = None
        if date is not None:
            dt = pd.to_datetime(date)
            cache_key = dt.strftime("%Y%m%d")
            if cache_key in self.cache:
                return self.cache[cache_key]
            start_dt = dt.replace(hour=0, minute=0, second=0).strftime("%Y%m%d%H%M%S")
            end_dt = dt.replace(hour=23, minute=59, second=59).strftime("%Y%m%d%H%M%S")
        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": page_size,
            "format": "json"
        }
        if date is not None:
            params["startdatetime"] = start_dt
            params["enddatetime"] = end_dt

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(self.base_url, params=params, headers=self.headers,
                                        timeout=10, verify=certifi.where())
                response.raise_for_status()
                data = response.json()
                articles = data.get("articles") or data.get("results") or []
                headlines = [article["title"] for article in articles if "title" in article]
                if not headlines:
                    logging.warning("No headlines found from GDELT; using fallback headlines.")
                    headlines = ["Bitcoin market update", "Crypto news: volatility persists"]
                if cache_key is not None:
                    self.cache[cache_key] = headlines
                return headlines
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    logging.warning("GDELT rate limit hit (HTTP 429). Attempt %d/%d.", attempt + 1, max_retries)
                else:
                    logging.error("HTTP error from GDELT: %s", e)
                    break
            except Exception as e:
                logging.error("Error fetching headlines from GDELT: %s", e)
                break
        logging.warning("Returning fallback headlines.")
        return ["Bitcoin market update", "Crypto news: volatility persists"]

class SentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert", device="cpu"):
        self.device = device
        logging.info("Loading FinBERT model for financial sentiment analysis (%s)...", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        # Use synchronous Google Translator via deep_translator.
        self.translator = GoogleTranslator(source='auto', target='en')

    def compute_sentiment(self, headlines):
        translations = []
        for headline in headlines:
            try:
                translated_text = self.translator.translate(headline)
                translations.append(translated_text)
            except Exception as e:
                logging.error("Error translating headline '%s': %s", headline, e)
                # Fallback: use the original headline if translation fails.
                translations.append(headline)
        
        scores = []
        for translated_text in translations:
            inputs = self.tokenizer(translated_text, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            sentiment = torch.softmax(outputs.logits, dim=-1)
            # FinBERT outputs [negative, neutral, positive]; compute score as (positive - negative).
            score = sentiment[0, 2].item() - sentiment[0, 0].item()
            scores.append(score)
        avg_score = np.mean(scores)
        logging.info("Average sentiment score: %.4f", avg_score)
        return avg_score
