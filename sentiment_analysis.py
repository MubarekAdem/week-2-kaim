"""
sentiment_analysis.py

Task 2 - Sentiment Analysis for Bank Reviews

Features:
- Loads processed reviews CSV (from Task 1)
- Computes sentiment label and score using:
    1) Hugging Face transformers pipeline with model:
       'distilbert-base-uncased-finetuned-sst-2-english'
    2) Fallback option: VADER (fast, rule-based)
- Aggregates results by bank and rating (basic KPIs)
- Saves `reviews_with_sentiment.csv` to data/processed
- Command-line arguments for input/output and batch size
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm
import math
import json

# Try imports for transformers; if not available we'll fallback to VADER
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# VADER fallback
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.data.find("tokenizers/punkt")
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


class SentimentAnalyzer:
    def __init__(self, method="transformers", model_name="distilbert-base-uncased-finetuned-sst-2-english", device=-1):
        """
        method: "transformers" or "vader"
        device: -1 for CPU, 0+ for GPU device index (transformers only)
        """
        self.method = method
        self.model_name = model_name
        self.device = device

        if self.method == "transformers":
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError(
                    "Transformers not available in this environment. Install 'transformers' or choose method='vader'.")
            # Initialize pipeline
            # Use tokenizer+model explicit loading to avoid cold-start surprises
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name)
            self.pipe = pipeline("sentiment-analysis", model=self.model,
                                 tokenizer=self.tokenizer, device=self.device)
        else:
            if not VADER_AVAILABLE:
                # Attempt to download NLTK data
                import nltk
                try:
                    nltk.download('vader_lexicon', quiet=True)
                    from nltk.sentiment.vader import SentimentIntensityAnalyzer
                except Exception as e:
                    raise RuntimeError(
                        "VADER not available and download failed. Install nltk and try again.") from e
            self.vader = SentimentIntensityAnalyzer()

    def analyze_batch_transformers(self, texts):
        """
        texts: list[str]
        returns list of dicts: {label: 'POSITIVE'|'NEGATIVE', score: float}
        """
        results = self.pipe(texts, truncation=True)
        # results are like [{"label":"POSITIVE","score":0.9995}, ...]
        return results

    def analyze_vader(self, texts):
        """
        VADER returns compound score in [-1,1]
        We'll map to label and normalized score [0,1]
        """
        out = []
        for t in texts:
            s = self.vader.polarity_scores(t)['compound']
            # label mapping
            if s >= 0.05:
                label = "POSITIVE"
            elif s <= -0.05:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            # normalize score to 0..1 for easier downstream aggregation
            score = (s + 1) / 2
            out.append({"label": label, "score": score})
        return out

    def analyze(self, texts, batch_size=32):
        """
        Generic analyze function that batches inputs.
        Returns list of dicts: {"label":..., "score":...}
        """
        results = []
        n = len(texts)
        for i in tqdm(range(0, n, batch_size), desc="Sentiment Batches"):
            batch = texts[i:i+batch_size]
            if self.method == "transformers":
                res = self.analyze_batch_transformers(batch)
                # convert labels and scores to consistent format
                for r in res:
                    # map model label (POSITIVE/NEGATIVE) to neutral handling absence
                    label = r.get("label", "NEUTRAL")
                    score = float(r.get("score", 0.0))
                    results.append({"label": label, "score": score})
            else:
                res = self.analyze_vader(batch)
                results.extend(res)
        return results


def aggregate_kpis(df, out_path=None):
    """
    Compute simple aggregation:
    - mean sentiment score by bank
    - mean sentiment score by bank & rating
    Save aggregate JSON if out_path provided
    """
    kpis = {}
    # numeric sentiment_score column must exist
    if 'sentiment_score' not in df.columns:
        raise ValueError("DataFrame must contain 'sentiment_score' column")

    # By bank
    bank_group = df.groupby('bank_name')['sentiment_score'].agg(
        ['count', 'mean', 'median', 'std']).reset_index()
    kpis['by_bank'] = bank_group.to_dict(orient='records')

    # By bank and rating
    br_group = df.groupby(['bank_name', 'rating'])[
        'sentiment_score'].agg(['count', 'mean']).reset_index()
    kpis['by_bank_rating'] = br_group.to_dict(orient='records')

    if out_path:
        ensure_dir(out_path)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(kpis, f, indent=2, ensure_ascii=False)

    return kpis


def run(args):
    # Load processed reviews CSV
    input_path = args.input
    output_path = args.output
    batch_size = args.batch_size
    method = args.method
    device = args.device

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    # Basic expectation: df should contain review_id, review_text, rating, bank_name
    required_cols = ['review_id', 'review_text', 'rating', 'bank_name']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in input CSV: {c}")

    # Instantiate analyzer
    print(f"Initializing sentiment analyzer: method={method}")
    analyzer = SentimentAnalyzer(method=method, device=device)

    # Process reviews in batches
    texts = df['review_text'].astype(str).tolist()
    results = analyzer.analyze(texts, batch_size=batch_size)

    # Attach results back to dataframe
    df['sentiment_label'] = [r['label'] for r in results]
    df['sentiment_score'] = [r['score'] for r in results]

    # Save per-review results CSV
    ensure_dir(output_path)
    df.to_csv(output_path, index=False)
    print(f"Saved sentiment results to: {output_path}")

    # Produce aggregate KPIs
    kpi_out = os.path.join(os.path.dirname(output_path), "sentiment_kpis.json")
    kpis = aggregate_kpis(df, out_path=kpi_out)
    print(f"Saved KPIs to: {kpi_out}")

    # Print sample aggregates (console)
    print("\nSample aggregate - mean sentiment by bank:")
    for row in kpis['by_bank']:
        print(
            f"  {row['bank_name']}: count={row['count']}, mean={row['mean']:.4f}")

    return df, kpis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis for Bank Reviews")
    parser.add_argument("--input", "-i", default="data/processed/reviews_processed.csv",
                        help="Path to processed reviews CSV")
    parser.add_argument("--output", "-o", default="data/processed/reviews_with_sentiment.csv",
                        help="Output path for CSV with sentiment")
    parser.add_argument("--method", "-m", choices=[
                        "transformers", "vader"], default="transformers", help="Sentiment method")
    parser.add_argument("--batch_size", "-b", type=int,
                        default=32, help="Batch size for model inference")
    parser.add_argument("--device", "-d", type=int, default=-1,
                        help="Device index for transformers (-1 for CPU, 0 for first GPU)")
    args = parser.parse_args()

    df_out, kpis = run(args)
