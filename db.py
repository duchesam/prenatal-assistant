import os
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def get_conn():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    due_date DATE NOT NULL DEFAULT '2027-02-05'
                );

                CREATE TABLE IF NOT EXISTS kick_logs (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT REFERENCES users(user_id),
                    movements INT NOT NULL,
                    minutes_elapsed INT NOT NULL,
                    logged_at TIMESTAMP DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS checklists (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT REFERENCES users(user_id),
                    category TEXT NOT NULL,
                    item TEXT NOT NULL,
                    UNIQUE(user_id, category, item)
                );
            """)
        conn.commit()


def ensure_user(user_id: str, due_date: str = "2027-02-05"):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (user_id, due_date)
                VALUES (%s, %s)
                ON CONFLICT (user_id) DO NOTHING;
                """,
                (user_id, due_date),
            )
        conn.commit()
