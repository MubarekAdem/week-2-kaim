import pandas as pd


def load_data(path="data/processed/reviews_final.csv"):
    return pd.read_csv(path)


def compute_sentiment_summary(df):
    return df.groupby(["bank_name", "sentiment_label"]).size().unstack(fill_value=0)


def compute_rating_distribution(df):
    return df.groupby(["bank_name", "rating"]).size().unstack(fill_value=0)


def compute_theme_counts(df):
    return df.groupby(["bank_name", "identified_theme(s)"]).size().unstack(fill_value=0)


def compute_drivers_and_painpoints(df):
    drivers = {}
    pain_points = {}

    banks = df["bank_name"].unique()

    for bank in banks:
        bank_df = df[df["bank_name"] == bank]

        # Themes are stored in "identified_theme(s)"
        top_positive = (bank_df[bank_df["sentiment_label"] == "POSITIVE"]
                        ["identified_theme(s)"].value_counts().head(2).to_dict())

        top_negative = (bank_df[bank_df["sentiment_label"] == "NEGATIVE"]
                        ["identified_theme(s)"].value_counts().head(2).to_dict())

        drivers[bank] = top_positive
        pain_points[bank] = top_negative

    return drivers, pain_points


def main():
    df = load_data()

    sentiment_summary = compute_sentiment_summary(df)
    rating_summary = compute_rating_distribution(df)
    themes = compute_theme_counts(df)
    drivers, pains = compute_drivers_and_painpoints(df)

    print("\n=== SENTIMENT SUMMARY ===")
    print(sentiment_summary)

    print("\n=== RATING SUMMARY ===")
    print(rating_summary)

    print("\n=== THEME COUNTS ===")
    print(themes)

    print("\n=== DRIVERS ===")
    print(drivers)

    print("\n=== PAIN POINTS ===")
    print(pains)


if __name__ == "__main__":
    main()
