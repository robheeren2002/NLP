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
import sqlite3
import streamlit as st

from datasets.telemetry import TelemetryLogger
from inputs import DEFAULT_SLIDERS, DEFAULT_RAG_SYSTEM, DEFAULT_RAG_HUMAN, DEFAULT_REPHRASE_SYSTEM, \
    DEFAULT_REPHRASE_HUMAN, DB_PATH, TELEMETRY_DB_PATH, generated_rag_system_prompt

# -----------------------------
# Session state
# -----------------------------
def ensure_state_defaults():
    st.session_state.setdefault("is_admin", False)
    st.session_state.setdefault("chain_version", 0)

    # Prompt texts used by home.py
    st.session_state.setdefault("rag_system_prompt", DEFAULT_RAG_SYSTEM)
    st.session_state.setdefault("rag_human_template", DEFAULT_RAG_HUMAN)
    st.session_state.setdefault("rephrase_system_prompt", DEFAULT_REPHRASE_SYSTEM)
    st.session_state.setdefault("rephrase_human_template", DEFAULT_REPHRASE_HUMAN)

    # Policy builder controls
    st.session_state.setdefault("price_policy", DEFAULT_SLIDERS["price_policy"])
    st.session_state.setdefault("response_format", DEFAULT_SLIDERS["response_format"])
    st.session_state.setdefault("response_length", DEFAULT_SLIDERS["response_length"])

    # Track whether the manual editor has diverged from the generated prompt
    st.session_state.setdefault("manual_overrides", False)

# -----------------------------
# Apply helpers
# -----------------------------
def apply_generated_from_toggles():
    """
    Overwrite the manual prompts with the prompt generated from current toggles,
    and rebuild chain.
    """
    st.session_state.rag_system_prompt = generated_rag_system_prompt(st.session_state.price_policy,
                                                                     st.session_state.response_format,
                                                                     st.session_state.response_length)
    # Keep templates (human/rephrase) stable; you can also generate them if you want.
    st.session_state.rag_human_template = DEFAULT_RAG_HUMAN
    st.session_state.rephrase_system_prompt = DEFAULT_REPHRASE_SYSTEM
    st.session_state.rephrase_human_template = DEFAULT_REPHRASE_HUMAN

    st.session_state.manual_overrides = False
    st.session_state.chain_version += 1


def apply_manual_edits():
    """
    Use whatever is currently in the manual editor fields and rebuild chain.
    """
    st.session_state.manual_overrides = True
    st.session_state.chain_version += 1


def reset_all_to_defaults():
    st.session_state.price_policy = "Only mention price when asked"
    st.session_state.response_format = "Full sentences"
    st.session_state.response_length = "Medium"
    apply_generated_from_toggles()


def current_slider_payload():
    return {
        "price_policy": st.session_state.get("price_policy") or DEFAULT_SLIDERS["price_policy"],
        "response_format": st.session_state.get("response_format") or DEFAULT_SLIDERS["response_format"],
        "response_length": st.session_state.get("response_length") or DEFAULT_SLIDERS["response_length"],
        "settings_version": st.session_state.get("chain_version", 0),  # NEW
    }


def all_known_sessions():
    # Pull sessions from message_store (history DB)
    if not os.path.exists(DB_PATH):
        return []
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    try:
        cur.execute("SELECT DISTINCT session_id FROM message_store WHERE session_id IS NOT NULL")
        rows = cur.fetchall()
    except Exception:
        rows = []
    con.close()
    return sorted([r[0] for r in rows if r and r[0]])

# -----------------------------
# Guard + UI
# -----------------------------
ensure_state_defaults()
if not st.session_state.get("is_admin", False):
    st.error("Not authorized.")
    st.stop()

st.title("🧠 Prompts & Strategy (Admin)")

st.markdown(
    "Use the controls below to generate a strategy **or** manually edit the prompts. "
    "When you apply toggles, the manual editor is updated to show the resulting prompts."
)

# --- Policy Builder ---
st.subheader("Strategy builder (toggles)")

st.session_state.price_policy = st.radio(
    "Price behavior",
    options=[
        "Always mention price",
        "Never mention price (visit site if available)",
        "Only mention price when asked",
    ],
    index=[
        "Always mention price",
        "Never mention price (visit site if available)",
        "Only mention price when asked",
    ].index(st.session_state.price_policy),
)

c1, c2 = st.columns(2)
with c1:
    st.session_state.response_format = st.radio(
        "Response format",
        options=["Full sentences", "Bullet points"],
        index=["Full sentences", "Bullet points"].index(st.session_state.response_format),
    )
with c2:
    st.session_state.response_length = st.radio(
        "Response length",
        options=["Short", "Medium", "Long"],
        index=["Short", "Medium", "Long"].index(st.session_state.response_length),
    )

# Show preview of what toggles would generate
with st.expander("Preview (generated system prompt from toggles)", expanded=False):
    st.code(generated_rag_system_prompt(st.session_state.price_policy, st.session_state.response_format,
                                        st.session_state.response_length), language="text")

all_sessions = all_known_sessions()
active_session = st.session_state.get("session_id")

scope = st.radio(
    "Apply to",
    options=["Active user", "Selected users", "All users"],
    horizontal=True,
)

selected = []
if scope == "Selected users":
    selected = st.multiselect("Choose users (session_id)", options=all_sessions, default=[active_session] if active_session in all_sessions else [])
elif scope == "All users":
    selected = all_sessions
else:
    selected = [active_session] if active_session else []

pb1, pb2, pb3 = st.columns([1, 1, 2])
with pb1:
    if st.button("Apply toggles → update prompts & rebuild"):
        # 1) Update admin_pages prompt fields + bump chain_version
        apply_generated_from_toggles()

        # 2) Persist sliders for chosen users (scope/selected)
        tel = TelemetryLogger(TELEMETRY_DB_PATH)
        tel.set_user_sliders([s for s in selected if s], current_slider_payload())

        st.success(
            f"Applied toggles and saved sliders for {len([s for s in selected if s])} user(s). "
            "They will take effect on the user's next interaction."
        )
with pb2:
    if st.button("Reset to defaults"):
        reset_all_to_defaults()
        st.success("Reset. Manual editor updated to defaults.")
with pb3:
    if st.session_state.manual_overrides:
        st.warning("Manual edits are currently active (may differ from toggle preview).")



st.divider()

# --- Manual editor ---
st.subheader("Manual prompt editor")

st.caption(
    "Edit these prompts directly. Click **Apply manual edits** to rebuild. "
    "If you later click **Apply toggles**, the system prompt will be overwritten by the generated version."
)

st.session_state.rag_system_prompt = st.text_area(
    "RAG system prompt",
    value=st.session_state.rag_system_prompt,
    height=260,
)

st.session_state.rag_human_template = st.text_area(
    "RAG human template",
    value=st.session_state.rag_human_template,
    height=240,
)

st.session_state.rephrase_system_prompt = st.text_area(
    "Rephrase system prompt",
    value=st.session_state.rephrase_system_prompt,
    height=160,
)

st.session_state.rephrase_human_template = st.text_area(
    "Rephrase human template",
    value=st.session_state.rephrase_human_template,
    height=80,
)

m1, m2, m3 = st.columns([1, 1, 2])
with m1:
    if st.button("Apply manual edits & rebuild"):
        apply_manual_edits()
        st.success("Applied manual edits. Home will rebuild on next question (or refresh).")
with m2:
    if st.button("Regenerate from toggles (overwrite manual)"):
        apply_generated_from_toggles()
        st.success("Overwritten manual prompts with toggle-generated prompts.")
with m3:
    st.info(f"chain_version = {st.session_state.chain_version}")
