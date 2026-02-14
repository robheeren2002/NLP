# ============================================================
# This code is created with the help of AI.
# All chats can be found at:
# https://chatgpt.com/share/699059a9-13a0-8013-ad55-189696111186
# https://chatgpt.com/share/69905ca0-85c8-8013-bc4f-14ad1864d1db
# https://chatgpt.com/share/69905d90-3474-8013-8db7-f87415685c7e
# ============================================================
import os, json, sqlite3
from datetime import datetime, timezone, timedelta

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def parse_iso_utc(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        # Handles "...+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None

class TelemetryLogger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as con:
            # ---- existing tables ----
            con.execute("""
            CREATE TABLE IF NOT EXISTS turns (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT UNIQUE,
              session_id TEXT,
              ts_utc TEXT,
              user_question_raw TEXT,
              user_question_rephrased TEXT,
              answer_text TEXT,
              prompt_templates_json TEXT,
              prompt_rendered_json TEXT,
              sliders_json TEXT,
              retrieved_docs_json TEXT,
              retrieved_source_ids_json TEXT,
              images_json TEXT,
              model_json TEXT,
              metrics_json TEXT
            );
            """)

            con.execute("""
            CREATE TABLE IF NOT EXISTS slider_settings (
              session_id TEXT PRIMARY KEY,
              sliders_json TEXT NOT NULL,
              updated_at_utc TEXT NOT NULL
            );
            """)

            # ---- NEW: per-user moderation state ----
            con.execute("""
            CREATE TABLE IF NOT EXISTS user_moderation (
              session_id TEXT PRIMARY KEY,
              guardrail_hits INTEGER NOT NULL DEFAULT 0,
              locked_until_utc TEXT,
              last_trigger_type TEXT,
              last_trigger_ts_utc TEXT,
              updated_at_utc TEXT NOT NULL
            );
            """)

            # ---- NEW: event log (one row per trigger) ----
            con.execute("""
            CREATE TABLE IF NOT EXISTS guardrail_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              ts_utc TEXT NOT NULL,
              trigger_type TEXT NOT NULL,
              question_excerpt TEXT,
              extra_json TEXT
            );
            """)

            con.commit()

    # -----------------------------
    # Existing slider methods
    # -----------------------------
    def get_user_sliders(self, session_id: str) -> dict | None:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute("SELECT sliders_json FROM slider_settings WHERE session_id=?", (session_id,))
            row = cur.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0] or "{}")
        except Exception:
            return None

    def set_user_sliders(self, session_ids: list[str], sliders: dict):
        payload = json.dumps(sliders, ensure_ascii=False, default=str)
        ts = _utc_now_iso()
        with self._connect() as con:
            for sid in session_ids:
                con.execute("""
                INSERT INTO slider_settings(session_id, sliders_json, updated_at_utc)
                VALUES(?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  sliders_json=excluded.sliders_json,
                  updated_at_utc=excluded.updated_at_utc
                """, (sid, payload, ts))
            con.commit()

    # -----------------------------
    # Existing telemetry logging
    # -----------------------------
    def log_turn(self, record: dict):
        def dumps(x):
            return json.dumps(x, ensure_ascii=False, default=str)

        row = (
            record.get("run_id"),
            record.get("session_id"),
            record.get("ts_utc") or utc_now_iso(),
            record.get("user_question_raw"),
            record.get("user_question_rephrased"),
            record.get("answer_text"),
            dumps(record.get("prompt_templates", {})),
            dumps(record.get("prompt_rendered", {})),
            dumps(record.get("sliders", {})),
            dumps(record.get("retrieved_docs", [])),
            dumps(record.get("retrieved_source_ids", [])),
            dumps(record.get("images", [])),
            dumps(record.get("model", {})),
            dumps(record.get("metrics", {})),
        )

        with self._connect() as con:
            con.execute("""
            INSERT OR REPLACE INTO turns (
              run_id, session_id, ts_utc,
              user_question_raw, user_question_rephrased,
              answer_text,
              prompt_templates_json, prompt_rendered_json, sliders_json,
              retrieved_docs_json, retrieved_source_ids_json, images_json,
              model_json, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, row)
            con.commit()

    # -----------------------------
    # NEW: moderation helpers
    # -----------------------------
    def get_user_moderation(self, session_id: str) -> dict:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute("""
                SELECT guardrail_hits, locked_until_utc, last_trigger_type, last_trigger_ts_utc
                FROM user_moderation
                WHERE session_id=?
            """, (session_id,))
            row = cur.fetchone()

        if not row:
            return {
                "session_id": session_id,
                "guardrail_hits": 0,
                "locked_until_utc": None,
                "last_trigger_type": None,
                "last_trigger_ts_utc": None,
            }

        hits, locked_until, last_type, last_ts = row
        return {
            "session_id": session_id,
            "guardrail_hits": int(hits or 0),
            "locked_until_utc": locked_until,
            "last_trigger_type": last_type,
            "last_trigger_ts_utc": last_ts,
        }

    def is_user_locked(self, session_id: str) -> tuple[bool, str | None]:
        st = self.get_user_moderation(session_id)
        locked_until = parse_iso_utc(st.get("locked_until_utc"))
        if locked_until and locked_until > datetime.now(timezone.utc):
            return True, locked_until.isoformat()
        return False, None

    def _upsert_user_moderation(self, session_id: str, hits: int, locked_until_utc: str | None,
                               last_trigger_type: str | None, last_trigger_ts_utc: str | None):
        ts = utc_now_iso()
        with self._connect() as con:
            con.execute("""
            INSERT INTO user_moderation(session_id, guardrail_hits, locked_until_utc, last_trigger_type, last_trigger_ts_utc, updated_at_utc)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
              guardrail_hits=excluded.guardrail_hits,
              locked_until_utc=excluded.locked_until_utc,
              last_trigger_type=excluded.last_trigger_type,
              last_trigger_ts_utc=excluded.last_trigger_ts_utc,
              updated_at_utc=excluded.updated_at_utc
            """, (session_id, int(hits), locked_until_utc, last_trigger_type, last_trigger_ts_utc, ts))
            con.commit()

    def log_guardrail_event(self, session_id: str, trigger_type: str, question_excerpt: str = "", extra: dict | None = None):
        ts = utc_now_iso()
        extra_json = json.dumps(extra or {}, ensure_ascii=False, default=str)
        with self._connect() as con:
            con.execute("""
                INSERT INTO guardrail_events(session_id, ts_utc, trigger_type, question_excerpt, extra_json)
                VALUES(?, ?, ?, ?, ?)
            """, (session_id, ts, trigger_type, (question_excerpt or "")[:500], extra_json))
            con.commit()

    def record_guardrail_hit(
        self,
        session_id: str,
        trigger_type: str,
        question_excerpt: str = "",
        *,
        lock_after: int = 3,
        lock_minutes: int = 15,
        extra: dict | None = None,
    ) -> dict:
        """
        Increment hit count and lock user if threshold reached.
        Returns updated moderation state.
        """
        st = self.get_user_moderation(session_id)
        hits = int(st.get("guardrail_hits") or 0) + 1

        now = datetime.now(timezone.utc)
        locked_until_utc = st.get("locked_until_utc")

        # lock if threshold reached (or re-lock if already locked and new hit occurs)
        if hits >= lock_after:
            locked_until = now + timedelta(minutes=int(lock_minutes))
            locked_until_utc = locked_until.isoformat()

        self.log_guardrail_event(session_id, trigger_type, question_excerpt=question_excerpt, extra=extra)

        self._upsert_user_moderation(
            session_id=session_id,
            hits=hits,
            locked_until_utc=locked_until_utc,
            last_trigger_type=trigger_type,
            last_trigger_ts_utc=now.isoformat(),
        )

        return self.get_user_moderation(session_id)

    def reset_user_moderation(self, session_id: str):
        """Optional admin_pages/dev helper."""
        self._upsert_user_moderation(session_id, hits=0, locked_until_utc=None, last_trigger_type=None, last_trigger_ts_utc=None)
