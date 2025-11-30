"""
thematic_analysis.py

Task: Extract themes from bank reviews using keyword-based grouping

Input: CSV with review_id, review_text, sentiment_label, sentiment_score
Output: CSV with additional column 'identified_theme(s)'
"""
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import re


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# Define themes and associated keywords
BANK_THEMES = {
    "Abyssinia Bank": {
        "Account Access Issues": ["login", "password", "access denied", "account blocked"],
        "Transaction Performance": ["slow transfer", "transaction failed", "delay"],
        "User Interface & Experience": ["app interface", "ui", "ux", "navigation"],
        "Customer Support": ["support", "helpdesk", "call center", "response"],
        "Feature Requests": ["feature", "option", "request", "wish"]
    },
    "Commercial Bank of Ethiopia": {
        "Account Access Issues": ["login", "password", "blocked", "access denied"],
        "Transaction Performance": ["slow", "transfer failed", "transaction delay"],
        "User Interface & Experience": ["ui", "app interface", "navigation", "experience"],
        "Customer Support": ["support", "call center", "response", "service"],
        "Feature Requests": ["request", "feature", "wish", "option"]
    },
    "Dashen Bank": {
        "Account Access Issues": ["login", "password", "access", "blocked"],
        "Transaction Performance": ["slow transfer", "failed transaction", "delay"],
        "User Interface & Experience": ["ui", "ux", "navigation", "interface"],
        "Customer Support": ["support", "service", "call center", "response"],
        "Feature Requests": ["feature", "option", "wish", "request"]
    }
}


def extract_themes(df):
    """
    For each review, check if keywords appear in review_text and assign themes
    Returns df with 'identified_theme(s)' column
    """
    themes_col = []

    for _, row in df.iterrows():
        review_text = str(row['review_text']).lower()
        bank = row['bank_name']
        bank_themes = BANK_THEMES.get(bank, {})
        matched_themes = []

        for theme_name, keywords in bank_themes.items():
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw.lower()) + r"\b", review_text):
                    matched_themes.append(theme_name)
                    break  # Avoid duplicate theme if multiple keywords match

        if not matched_themes:
            matched_themes = ["Other"]  # Default theme if nothing matches

        themes_col.append(", ".join(matched_themes))

    df['identified_theme(s)'] = themes_col
    return df


def run(args):
    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = ['review_id', 'review_text',
                     'sentiment_label', 'sentiment_score', 'bank_name']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in input CSV: {c}")

    df = extract_themes(df)

    ensure_dir(output_path)
    df.to_csv(output_path, index=False)
    print(f"Saved themed reviews to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thematic Analysis for Bank Reviews")
    parser.add_argument("--input", "-i", default="data/processed/reviews_with_sentiment_vader.csv",
                        help="Path to CSV with sentiment scores")
    parser.add_argument("--output", "-o", default="data/processed/reviews_with_themes.csv",
                        help="Output path for CSV with identified themes")
    args = parser.parse_args()

    run(args)
