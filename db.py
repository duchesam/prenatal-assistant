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
                    due_date DATE NOT NULL DEFAULT '2027-02-05',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kick_logs (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE,
                    movements INTEGER NOT NULL,
                    minutes_elapsed INTEGER NOT NULL,
                    logged_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS checklists (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE,
                    category TEXT NOT NULL,
                    item TEXT NOT NULL,
                    UNIQUE(user_id, category, item)
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT REFERENCES users(user_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
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

def save_message(user_id: str, role: str, content: str):
    ensure_user(user_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_messages (user_id, role, content)
                VALUES (%s, %s, %s);
                """,
                (user_id, role, content),
            )
        conn.commit()

def load_messages(user_id: str, limit: int = 50) -> list[dict]:
    ensure_user(user_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM (
                    SELECT id, role, content
                    FROM chat_messages
                    WHERE user_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                ) sub
                ORDER BY id ASC;
                """,
                (user_id, limit),
            )
            return [{"role": r["role"], "content": r["content"]} for r in cur.fetchall()]

def clear_messages(user_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_messages WHERE user_id = %s;", (user_id,))
        conn.commit()
