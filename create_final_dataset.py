import pandas as pd

# Load all datasets
df_clean = pd.read_csv("data/processed/reviews_processed.csv")
df_sent = pd.read_csv("data/processed/reviews_with_sentiment_vader.csv")
df_themes = pd.read_csv("data/processed/reviews_with_themes.csv")

# 1. Merge sentiment with clean dataset
df_final = df_clean.merge(
    df_sent[['review_id', 'sentiment_label', 'sentiment_score']],
    on="review_id",
    how="left"
)

# 2. Merge themes (identified_theme(s))
df_final = df_final.merge(
    df_themes[['review_id', 'identified_theme(s)']],
    on="review_id",
    how="left"
)

# Save result
df_final.to_csv("data/processed/reviews_final.csv", index=False)

print("✔ Final dataset created → data/processed/reviews_final.csv")
print("Total rows:", len(df_final))
