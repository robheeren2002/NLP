# ============================================================
# This code is created with the help of AI. As a starting
# point, there was already a working code where a RAG chain
# could be invoked. ChatGPT 5.2 was used to craft a streamlit
# application around this code. The chats can be found at:
# https://chatgpt.com/share/6990586a-0d68-8013-880f-243ee001d006
# https://chatgpt.com/share/69905d90-3474-8013-8db7-f87415685c7e
# ============================================================
import os
import json
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from inputs import QUESTIONNAIRE_DB_PATH

# -----------------------------
# Guard (admin_pages only)
# -----------------------------
if not st.session_state.get("is_admin", False):
    st.error("Not authorized.")
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def json_loads_safe(x, default=None):
    if x is None:
        return default
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return default


def flatten(prefix: str, d: dict) -> dict:
    """Flatten one level: {"a":1,"b":{"c":2}} -> {"a":1,"b.c":2} (only for dict values)."""
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{prefix}{k}.{kk}"] = vv
        else:
            out[f"{prefix}{k}"] = v
    return out


# -----------------------------
# Load surveys (cached)
# -----------------------------
@st.cache_data(show_spinner=True)
def load_user_surveys(db_path: str) -> pd.DataFrame:
    if not db_path or not os.path.exists(db_path):
        return pd.DataFrame()

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        # Ensure table exists; if not, return empty
        try:
            result = conn.execute(text("SELECT * FROM user_surveys ORDER BY updated_at_utc DESC"))
        except Exception:
            return pd.DataFrame()

        rows = result.fetchall()
        cols = result.keys()

    df = pd.DataFrame.from_records([dict(zip(cols, r)) for r in rows])
    if df.empty:
        return df

    # Parse json columns
    df["pre"] = df.get("pre_json").apply(lambda x: json_loads_safe(x, None))
    df["post"] = df.get("post_json").apply(lambda x: json_loads_safe(x, None))

    # Convenience flags
    df["pre_completed"] = df["pre"].apply(lambda x: bool(x))
    df["post_completed"] = df["post"].apply(lambda x: bool(x))

    # Parse timestamps to datetime for sorting/filtering (optional)
    for c in ["pre_completed_at_utc", "post_completed_at_utc", "updated_at_utc"]:
        if c in df.columns:
            df[c + "_dt"] = pd.to_datetime(df[c], errors="coerce", utc=True)

    # Flatten common fields for easier filtering/table display
    flat_rows = []
    for _, r in df.iterrows():
        pre = r.get("pre") if isinstance(r.get("pre"), dict) else {}
        post = r.get("post") if isinstance(r.get("post"), dict) else {}
        flat = {}
        flat.update(flatten("pre.", pre))
        flat.update(flatten("post.", post))
        flat_rows.append(flat)

    df_flat = pd.DataFrame(flat_rows)
    df = pd.concat([df, df_flat], axis=1)

    return df


# -----------------------------
# UI
# -----------------------------
st.title("🧾 Surveys (Admin)")

col_a, col_b = st.columns([1, 4])
with col_a:
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

db_path = QUESTIONNAIRE_DB_PATH
st.caption(f"DB: {db_path}")

df = load_user_surveys(db_path)

if df.empty:
    st.info("No survey rows found (or table missing).")
    st.stop()


# ---- Filters ----
st.subheader("Filters")

all_sessions = sorted([s for s in df["session_id"].dropna().unique()]) if "session_id" in df.columns else []
default_session = st.session_state.get("session_id")
default_selection = [default_session] if default_session in all_sessions else []

selected_sessions = st.multiselect(
    "session_id",
    options=all_sessions,
    default=default_selection,
)

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    pre_filter = st.selectbox("Pre-survey", ["Any", "Completed", "Missing"], index=0)
with c2:
    post_filter = st.selectbox("Post-survey", ["Any", "Completed", "Missing"], index=0)
with c3:
    query = st.text_input("Search in flattened fields (pre.* / post.*)", value="")

last_n = st.number_input("Show last N", min_value=1, max_value=50000, value=500, step=100)

f = df.copy()

if selected_sessions and "session_id" in f.columns:
    f = f[f["session_id"].isin(selected_sessions)]

if pre_filter == "Completed":
    f = f[f["pre_completed"] == True]
elif pre_filter == "Missing":
    f = f[f["pre_completed"] == False]

if post_filter == "Completed":
    f = f[f["post_completed"] == True]
elif post_filter == "Missing":
    f = f[f["post_completed"] == False]

if query.strip():
    # Search across flattened columns only (pre./post.)
    flat_cols = [c for c in f.columns if c.startswith("pre.") or c.startswith("post.")]
    if flat_cols:
        mask = False
        q = query.strip()
        for c in flat_cols:
            mask = mask | f[c].astype(str).str.contains(q, case=False, na=False)
        f = f[mask]

# Default sort: updated_at_utc_dt desc if present
if "updated_at_utc_dt" in f.columns:
    f = f.sort_values("updated_at_utc_dt", ascending=False)
elif "updated_at_utc" in f.columns:
    f = f.sort_values("updated_at_utc", ascending=False)

f = f.head(int(last_n))


# ---- Sorting ----
st.subheader("Table")

sort_candidates = [c for c in ["session_id", "pre_completed", "post_completed", "updated_at_utc", "pre_completed_at_utc", "post_completed_at_utc"] if c in f.columns]
# Add some flattened keys too
sort_candidates += [c for c in f.columns if (c.startswith("pre.") or c.startswith("post."))][:50]

s1, s2 = st.columns([2, 2])
with s1:
    sort_col = st.selectbox("Sort column", options=sort_candidates, index=0 if sort_candidates else 0)
with s2:
    sort_dir = st.selectbox("Direction", ["Descending", "Ascending"], index=0)

ascending = sort_dir == "Ascending"
if sort_col in f.columns:
    f = f.sort_values(sort_col, ascending=ascending)


# ---- Visible columns ----
default_cols = [c for c in ["session_id", "pre_completed", "post_completed", "pre_completed_at_utc", "post_completed_at_utc", "updated_at_utc"] if c in f.columns]
# Include some common flattened columns if present
for cand in ["pre.dob", "pre.education", "pre.profession", "pre.ai_knowledge", "post.experience", "post.recommend", "post.purchased"]:
    if cand in f.columns and cand not in default_cols:
        default_cols.append(cand)

excluded_cols = [c for c in ["pre_json", "post_json", "pre", "post"] if c in f.columns]
other_cols = [c for c in f.columns if c not in default_cols and c not in excluded_cols]

visible_cols = st.multiselect(
    "Columns to display",
    options=default_cols + other_cols,
    default=default_cols,
)

table_df = f[visible_cols].copy() if visible_cols else f.copy()

st.dataframe(table_df, use_container_width=True, hide_index=True)


# ---- Download ----
export_df = table_df.copy()
csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv, file_name="user_surveys_filtered.csv", mime="text/csv")


# ---- Detail view ----
st.subheader("Row details")

if "session_id" in f.columns and len(f):
    selected_sid = st.selectbox(
        "Select session_id",
        options=f["session_id"].tolist(),
        index=0,
    )
    row = f[f["session_id"] == selected_sid].iloc[0].to_dict()

    with st.expander("Show selected row (parsed)", expanded=True):
        st.markdown(f"**session_id:** {row.get('session_id')}")
        st.markdown(f"**pre_completed_at_utc:** {row.get('pre_completed_at_utc')}")
        st.markdown(f"**post_completed_at_utc:** {row.get('post_completed_at_utc')}")
        st.markdown(f"**updated_at_utc:** {row.get('updated_at_utc')}")

        st.markdown("**pre (parsed):**")
        st.code(json.dumps(json_loads_safe(row.get("pre_json"), {}), ensure_ascii=False, indent=2, default=str), language="json")

        st.markdown("**post (parsed):**")
        st.code(json.dumps(json_loads_safe(row.get("post_json"), {}), ensure_ascii=False, indent=2, default=str), language="json")
