import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/reviews_final.csv")

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="bank", hue="sentiment_label")
plt.title("Sentiment Distribution per Bank")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/sentiment_distribution.png")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="rating", hue="bank")
plt.title("Rating Distribution per Bank")
plt.savefig("reports/rating_distribution.png")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="theme", hue="bank")
plt.title("Theme Frequency Comparison")
plt.xticks(rotation=45)
plt.savefig("reports/theme_frequency.png")
