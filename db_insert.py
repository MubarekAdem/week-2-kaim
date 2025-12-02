"""
Task 3: Insert Cleaned Reviews into PostgreSQL
"""

from config import DATA_PATHS
from dotenv import load_dotenv
import pandas as pd
import psycopg2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


load_dotenv()


class PostgresUploader:

    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            port=os.getenv("POSTGRES_PORT")
        )
        self.cursor = self.conn.cursor()

    def load_processed_data(self):
        print("Loading processed data...")
        df = pd.read_csv(DATA_PATHS["processed_reviews"])
        print(f"Loaded {len(df)} processed review records.")
        return df

    def insert_banks(self, df):
        print("Inserting banks...")
        bank_names = df[['bank_code', 'bank_name']].drop_duplicates()

        bank_id_map = {}

        for _, row in bank_names.iterrows():
            self.cursor.execute(
                """
                INSERT INTO banks (bank_name, app_name)
                VALUES (%s, %s)
                RETURNING bank_id;
                """,
                (row["bank_name"], row["bank_code"])
            )
            bank_id = self.cursor.fetchone()[0]
            bank_id_map[row["bank_code"]] = bank_id

        self.conn.commit()
        print("Bank records inserted successfully.")
        return bank_id_map

    def insert_reviews(self, df, bank_id_map):
        print("Inserting reviews into PostgreSQL...")

        count = 0
        for _, row in df.iterrows():
            try:
                self.cursor.execute(
                    """
                    INSERT INTO reviews (
                        review_id, bank_id, review_text, rating, review_date,
                        sentiment_label, sentiment_score, source
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (review_id) DO NOTHING;
                    """,
                    (
                        row["review_id"],
                        bank_id_map[row["bank_code"]],
                        row["review_text"],
                        row["rating"],
                        row["review_date"],
                        row.get("sentiment_label", None),
                        row.get("sentiment_score", None),
                        row.get("source", "Google Play")
                    )
                )
                count += 1
            except Exception as e:
                print(f"Error inserting review: {e}")

        self.conn.commit()
        print(f"{count} reviews successfully inserted.")

    def run(self):
        df = self.load_processed_data()
        bank_map = self.insert_banks(df)
        self.insert_reviews(df, bank_map)
        print("Task 3 completed successfully!")


if __name__ == "__main__":
    uploader = PostgresUploader()
    uploader.run()
