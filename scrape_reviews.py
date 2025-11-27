import nltk

# Download once (safe even if files already exist)
nltk.download('vader_lexicon')
nltk.download('stopwords')

from google_play_scraper import reviews, Sort
import pandas as pd

def scrape(app_id, count=400):
    result, _ = reviews(
        app_id,
        lang="en",
        country="us",
        sort=Sort.NEWEST,
        count=count
    )
    df = pd.DataFrame(result)
    df["app"] = app_id
    return df[["content", "score", "at", "app"]]

bank_apps = {
    "CBE": "com.combanketh.mobilebanking",
    "BOA": "com.boa.boaMobileBanking",
    "Dashen": "com.cr2.amolelight"
}

final_df = pd.concat([scrape(app_id) for app_id in bank_apps.values()])
final_df.to_csv("bank_reviews.csv", index=False)

print("Scraping complete! Saved to bank_reviews.csv")
