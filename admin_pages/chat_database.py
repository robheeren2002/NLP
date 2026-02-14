# ============================================================
# This code is created with the help of AI. As a starting
# point, there was already a working code where a RAG chain
# could be invoked. ChatGPT 5.2 was used to craft a streamlit
# application around this code. The chats can be found at:
# https://chatgpt.com/share/6990586a-0d68-8013-880f-243ee001d006
# https://chatgpt.com/share/699059a9-13a0-8013-ad55-189696111186
# https://chatgpt.com/share/69905ca0-85c8-8013-bc4f-14ad1864d1db
# ============================================================
import os
import json
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from inputs import DB_PATH, TELEMETRY_DB_PATH

# -----------------------------
# Guard (admin_pages only)
# -----------------------------
if not st.session_state.get("is_admin", False):
    st.error("Not authorized.")
    st.stop()


# -----------------------------
# Load + parse message_store (cached)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_message_store(db_path: str) -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM message_store ORDER BY id ASC"))
        rows = result.fetchall()
        cols = result.keys()

    records = []
    for r in rows:
        rec = dict(zip(cols, r))

        # Find the JSON column
        json_col = None
        for cand in ["message", "msg", "value", "data", "payload"]:
            if cand in rec:
                json_col = cand
                break
        if json_col is None and len(cols) >= 3:
            json_col = cols[2]

        raw = rec.get(json_col)
        parsed = {}
        if raw:
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {}

        msg_type = parsed.get("type")
        msg_data = parsed.get("data", {}) if isinstance(parsed, dict) else {}

        content = msg_data.get("content")
        name = msg_data.get("name")
        msg_id = msg_data.get("id")
        additional_kwargs = msg_data.get("additional_kwargs") or {}
        response_metadata = msg_data.get("response_metadata") or {}

        # Pull run_id/images_abs from additional_kwargs (when present)
        rec["run_id"] = additional_kwargs.get("run_id")
        rec["images_abs"] = additional_kwargs.get("images_abs")

        rec["_raw_json"] = raw
        rec["role"] = msg_type
        rec["content"] = content
        rec["name"] = name
        rec["msg_id"] = msg_id
        rec["_has_additional_kwargs"] = bool(additional_kwargs)
        rec["_has_response_metadata"] = bool(response_metadata)

        records.append(rec)

    df = pd.DataFrame.from_records(records)

    # Normalize session column name
    if "session_id" not in df.columns:
        for cand in ["session", "sessionId", "sid"]:
            if cand in df.columns:
                df.rename(columns={cand: "session_id"}, inplace=True)

    return df


# -----------------------------
# Load telemetry.turns (cached)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_telemetry_turns(db_path: str) -> pd.DataFrame:
    if not db_path or not os.path.exists(db_path):
        return pd.DataFrame()

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM turns ORDER BY ts_utc ASC"))
        rows = result.fetchall()
        cols = result.keys()

    df = pd.DataFrame.from_records([dict(zip(cols, r)) for r in rows])
    if df.empty:
        return df

    if "ts_utc" in df.columns:
        df["ts_utc_dt"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    else:
        df["ts_utc_dt"] = pd.NaT

    # Parse sliders_json
    if "sliders_json" in df.columns:
        df["sliders"] = df["sliders_json"].apply(lambda x: json_loads_safe(x, {}))
    else:
        df["sliders"] = [{} for _ in range(len(df))]

    return df

def json_loads_safe(x, default):
    if x is None:
        return default
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return default


def attach_telemetry_to_messages(df_msg: pd.DataFrame, df_tel: pd.DataFrame) -> pd.DataFrame:
    """
    Attach telemetry by run_id if available.
    Fallback: match assistant messages to telemetry by sequence within session_id.
    """
    df_msg = df_msg.copy()
    df_msg["telemetry"] = None

    if df_msg.empty or df_tel.empty:
        return df_msg

    # Build map: run_id -> telemetry row dict
    tel_by_run = {}
    if "run_id" in df_tel.columns:
        for _, r in df_tel.iterrows():
            rid = r.get("run_id")
            if rid:
                d = r.to_dict()
                d.pop("ts_utc_dt", None)
                tel_by_run[rid] = d

    # Attach by run_id where possible
    has_run = df_msg.get("run_id")
    if has_run is not None:
        for i, row in df_msg.iterrows():
            rid = row.get("run_id")
            if rid and rid in tel_by_run:
                df_msg.at[i, "telemetry"] = tel_by_run[rid]

    # Fallback sequence match for assistant rows without telemetry
    is_assistant = df_msg["role"].astype(str).isin(["ai", "assistant"])
    needs_fallback = is_assistant & df_msg["telemetry"].isna()

    if needs_fallback.any() and "session_id" in df_msg.columns and "session_id" in df_tel.columns:
        tel_by_session = {sid: g.copy() for sid, g in df_tel.groupby("session_id")}

        for sid, idxs in df_msg[needs_fallback].groupby("session_id").groups.items():
            tel = tel_by_session.get(sid)
            if tel is None or tel.empty:
                continue

            tel = tel.sort_values("ts_utc_dt") if "ts_utc_dt" in tel.columns else tel
            tel_records = [r.to_dict() for _, r in tel.iterrows()]
            for rec in tel_records:
                rec.pop("ts_utc_dt", None)

            msg_rows = df_msg.loc[idxs].copy()
            msg_rows = msg_rows.sort_values("id") if "id" in msg_rows.columns else msg_rows

            for n, ridx in enumerate(msg_rows.index.tolist()):
                if n >= len(tel_records):
                    break
                df_msg.at[ridx, "telemetry"] = tel_records[n]

    return df_msg


# -----------------------------
# UI
# -----------------------------
st.title("🗄️ Database (Admin)")

col_a, col_b = st.columns([1, 4])
with col_a:
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

df = load_message_store(DB_PATH)
df_tel = load_telemetry_turns(TELEMETRY_DB_PATH)
df = attach_telemetry_to_messages(df, df_tel)

if df.empty:
    st.info("No rows found.")
    st.stop()

# ---- Expand sliders into columns ----
def get_sliders_dict(tel):
    if not isinstance(tel, dict):
        return {}
    s = tel.get("sliders")
    if isinstance(s, dict):
        return s
    # fallback if older telemetry rows only have sliders_json
    return json_loads_safe(tel.get("sliders_json"), {})

# collect slider keys across current filtered dataset
all_slider_keys = set()
for tel in df["telemetry"].dropna():
    all_slider_keys.update(get_sliders_dict(tel).keys())

all_slider_keys = sorted(all_slider_keys)

for k in all_slider_keys:
    df[f"{k}"] = df["telemetry"].apply(lambda t: get_sliders_dict(t).get(k))

for k in all_slider_keys:
    df[f"{k}"] = df["telemetry"].apply(lambda t: get_sliders_dict(t).get(k))

slider_cols = all_slider_keys  # because you named the columns exactly as the keys

if slider_cols and "session_id" in df.columns and "role" in df.columns and "id" in df.columns:
    df = df.sort_values(["session_id", "id"]).copy()

    is_assistant = df["role"].astype(str).isin(["ai", "assistant"])

    # Only keep sliders on assistant rows (telemetry lives there)
    for c in slider_cols:
        df.loc[~is_assistant, c] = None

    # Backfill within each session: human row gets next assistant's slider values
    df[slider_cols] = df.groupby("session_id", dropna=False)[slider_cols].bfill()



# Add convenience telemetry columns
def safe_get(d, k, default=None):
    return d.get(k, default) if isinstance(d, dict) else default


df["telemetry_run_id"] = df["telemetry"].apply(lambda d: safe_get(d, "run_id"))
df["telemetry_ts_utc"] = df["telemetry"].apply(lambda d: safe_get(d, "ts_utc"))
df["telemetry_q_raw"] = df["telemetry"].apply(lambda d: safe_get(d, "user_question_raw"))
df["telemetry_q_rephrased"] = df["telemetry"].apply(lambda d: safe_get(d, "user_question_rephrased"))


# ---- Filters ----
st.subheader("Filters")

all_sessions = sorted([s for s in df["session_id"].dropna().unique()]) if "session_id" in df.columns else []
default_session = st.session_state.get("session_id")
default_selection = [default_session] if default_session in all_sessions else (all_sessions[:1] if all_sessions else [])

selected_sessions = st.multiselect(
    "session_id",
    options=all_sessions,
    default=default_selection,
)

roles = sorted([r for r in df["role"].dropna().unique()]) if "role" in df.columns else []
selected_roles = st.multiselect("role", options=roles, default=roles)

query = st.text_input("Search in content", value="")
last_n = st.number_input("Show last N", min_value=1, max_value=5000, value=200, step=50)

f = df.copy()

if selected_sessions and "session_id" in f.columns:
    f = f[f["session_id"].isin(selected_sessions)]

if selected_roles and "role" in f.columns:
    f = f[f["role"].isin(selected_roles)]

if query.strip():
    f = f[f["content"].fillna("").str.contains(query, case=False, na=False)]

if "id" in f.columns:
    f = f.sort_values("id").tail(int(last_n))


# ---- Sorting ----
st.subheader("Table")

s1, s2, s3 = st.columns([2, 2, 3])
with s1:
    sort_col = st.selectbox(
        "Sort column",
        options=[c for c in ["id", "session_id", "role"] if c in f.columns]
                + [c for c in f.columns if c not in ["_raw_json"]],
        index=0,
    )
with s2:
    sort_dir = st.selectbox("Direction", ["Ascending", "Descending"], index=1)

ascending = sort_dir == "Ascending"
if sort_col in f.columns:
    f = f.sort_values(sort_col, ascending=ascending)


# ---- Visible columns ----
default_cols = [c for c in ["id", "session_id", "role", "content"] if c in
                f.columns]
default_cols += [c for c in ["price_policy", "response_format", "response_length"] if c in
                f.columns]
excluded_cols = [c for c in ["message", "run_id", "images_abs", "name", "msg_id", "_has_additional_kwargs",
                             "_has_response_metadata","telemetry","telemetry_run_id", "telemetry_ts_utc",
                             "settings_version"] if c in
                 f.columns]
other_cols = [c for c in f.columns if c not in default_cols and c != "_raw_json" and c not in excluded_cols]

visible_cols = st.multiselect(
    "Columns to display",
    options=default_cols + other_cols,
    default=default_cols,
)

table_df = f[visible_cols].copy() if visible_cols else f.copy()

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
)
# ---- Download ----
export_df = table_df.copy()
if "content" in export_df.columns:
    export_df["content"] = (
        export_df["content"]
        .fillna("")
        .astype(str)
        .str.replace("\r\n", "\n", regex=False)
        .str.replace("\n", "\\n", regex=False)
    )

csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv, file_name="message_store_filtered.csv", mime="text/csv")


# ---- Detail view ----
st.subheader("Row details")

if "id" in f.columns and len(f):
    selected_id = st.selectbox(
        "Select row by id",
        options=f["id"].tolist(),
        index=len(f) - 1,
    )
    row = f[f["id"] == selected_id].iloc[0].to_dict()

    with st.expander("Show selected row (parsed)", expanded=True):
        st.markdown(f"**id:** {row.get('id')}")
        st.markdown(f"**session_id:** {row.get('session_id')}")
        st.markdown(f"**role:** {row.get('role')}")
        st.markdown(f"**run_id (from message additional_kwargs):** {row.get('run_id')}")
        st.markdown("**content:**")
        st.write(row.get("content"))

        imgs = row.get("images_abs")
        if isinstance(imgs, list) and imgs:
            st.markdown("**images_abs:**")
            st.code(json.dumps(imgs, ensure_ascii=False, indent=2), language="json")

        st.markdown("**raw JSON:**")
        st.code(row.get("_raw_json") or "", language="json")

        st.markdown("**telemetry (parsed):**")
        tel = row.get("telemetry") or {}
        st.code(json.dumps(tel, ensure_ascii=False, indent=2, default=str), language="json")


