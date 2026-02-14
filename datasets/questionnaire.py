# ============================================================
# This code is created with the help of AI.
# The chats can be found at:
# https://chatgpt.com/share/6990586a-0d68-8013-880f-243ee001d006
# https://chatgpt.com/share/69905d90-3474-8013-8db7-f87415685c7e
# ============================================================
import os, json, sqlite3
from datetime import datetime, timezone

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

class QuestionnaireLogger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as con:
            con.execute("""
            CREATE TABLE IF NOT EXISTS user_surveys (
              session_id TEXT PRIMARY KEY,
              pre_json TEXT,
              post_json TEXT,
              pre_completed_at_utc TEXT,
              post_completed_at_utc TEXT,
              updated_at_utc TEXT NOT NULL
            );
            """)
            con.commit()
            con.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
              session_id TEXT PRIMARY KEY,
              profile_text TEXT NOT NULL,
              source TEXT,
              updated_at_utc TEXT NOT NULL
            );
            """)

    # methods
    def get_user_survey(self, session_id: str) -> dict | None:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                "SELECT pre_json, post_json, pre_completed_at_utc, post_completed_at_utc "
                "FROM user_surveys WHERE session_id=?",
                (session_id,),
            )
            row = cur.fetchone()
        if not row:
            return None

        pre_json, post_json, pre_ts, post_ts = row
        try:
            pre = json.loads(pre_json) if pre_json else None
        except Exception:
            pre = None
        try:
            post = json.loads(post_json) if post_json else None
        except Exception:
            post = None

        return {"pre": pre, "post": post, "pre_completed_at_utc": pre_ts, "post_completed_at_utc": post_ts}

    def upsert_pre_survey(self, session_id: str, pre: dict):
        ts = utc_now_iso()
        payload = json.dumps(pre or {}, ensure_ascii=False, default=str)
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO user_surveys(session_id, pre_json, pre_completed_at_utc, updated_at_utc)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  pre_json=excluded.pre_json,
                  pre_completed_at_utc=excluded.pre_completed_at_utc,
                  updated_at_utc=excluded.updated_at_utc
                """,
                (session_id, payload, ts, ts),
            )
            con.commit()

    def upsert_post_survey(self, session_id: str, post: dict):
        ts = utc_now_iso()
        payload = json.dumps(post or {}, ensure_ascii=False, default=str)
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO user_surveys(session_id, post_json, post_completed_at_utc, updated_at_utc)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  post_json=excluded.post_json,
                  post_completed_at_utc=excluded.post_completed_at_utc,
                  updated_at_utc=excluded.updated_at_utc
                """,
                (session_id, payload, ts, ts),
            )
            con.commit()

    def get_user_profile(self, session_id: str) -> str | None:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute("SELECT profile_text FROM user_profiles WHERE session_id=?", (session_id,))
            row = cur.fetchone()
        return row[0] if row else None

    def upsert_user_profile(self, session_id: str, profile_text: str, source: str = "manual"):
        ts = utc_now_iso()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO user_profiles(session_id, profile_text, source, updated_at_utc)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  profile_text=excluded.profile_text,
                  source=excluded.source,
                  updated_at_utc=excluded.updated_at_utc
                """,
                (session_id, profile_text, source, ts),
            )
            con.commit()


