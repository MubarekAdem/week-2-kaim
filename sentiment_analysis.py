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
import json
import pandas as pd
import os
import argparse
from tqdm import tqdm

# --- VADER (only one import) ---
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)

# Transformers (optional)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


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
                    "Transformers not available. Install 'transformers' or choose method='vader'.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name)
            self.pipe = pipeline("sentiment-analysis", model=self.model,
                                 tokenizer=self.tokenizer, device=self.device)
        else:
            # Initialize VADER
            self.vader = SentimentIntensityAnalyzer()

    def analyze_batch_transformers(self, texts):
        results = self.pipe(texts, truncation=True)
        return results

    def analyze_vader(self, texts):
        out = []
        for t in texts:
            s = self.vader.polarity_scores(t)['compound']
            if s >= 0.05:
                label = "POSITIVE"
            elif s <= -0.05:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            score = (s + 1) / 2
            out.append({"label": label, "score": score})
        return out

    def analyze(self, texts, batch_size=32):
        results = []
        n = len(texts)
        for i in tqdm(range(0, n, batch_size), desc="Sentiment Batches"):
            batch = texts[i:i+batch_size]
            if self.method == "transformers":
                res = self.analyze_batch_transformers(batch)
                for r in res:
                    label = r.get("label", "NEUTRAL")
                    score = float(r.get("score", 0.0))
                    results.append({"label": label, "score": score})
            else:
                res = self.analyze_vader(batch)
                results.extend(res)
        return results


def aggregate_kpis(df, out_path=None):
    kpis = {}
    if 'sentiment_score' not in df.columns:
        raise ValueError("DataFrame must contain 'sentiment_score' column")

    bank_group = df.groupby('bank_name')['sentiment_score'].agg(
        ['count', 'mean', 'median', 'std']).reset_index()
    kpis['by_bank'] = bank_group.to_dict(orient='records')

    br_group = df.groupby(['bank_name', 'rating'])[
        'sentiment_score'].agg(['count', 'mean']).reset_index()
    kpis['by_bank_rating'] = br_group.to_dict(orient='records')

    if out_path:
        ensure_dir(out_path)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(kpis, f, indent=2, ensure_ascii=False)

    return kpis


def run(args):
    input_path = args.input
    output_path = args.output
    batch_size = args.batch_size
    method = args.method
    device = args.device

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    required_cols = ['review_id', 'review_text', 'rating', 'bank_name']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in input CSV: {c}")

    print(f"Initializing sentiment analyzer: method={method}")
    analyzer = SentimentAnalyzer(method=method, device=device)

    texts = df['review_text'].astype(str).tolist()
    results = analyzer.analyze(texts, batch_size=batch_size)

    df['sentiment_label'] = [r['label'] for r in results]
    df['sentiment_score'] = [r['score'] for r in results]

    ensure_dir(output_path)
    df.to_csv(output_path, index=False)
    print(f"Saved sentiment results to: {output_path}")

    kpi_out = os.path.join(os.path.dirname(output_path), "sentiment_kpis.json")
    kpis = aggregate_kpis(df, out_path=kpi_out)
    print(f"Saved KPIs to: {kpi_out}")

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
