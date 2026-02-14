# ============================================================
# This code is created with the help of AI. As a starting
# point, there was already a working code where a RAG chain
# could be invoked. ChatGPT 5.2 was used to craft a streamlit
# application around this code. The chats can be found at:
# https://chatgpt.com/share/6990586a-0d68-8013-880f-243ee001d006
# https://chatgpt.com/share/699059a9-13a0-8013-ad55-189696111186
# https://chatgpt.com/share/69905ca0-85c8-8013-bc4f-14ad1864d1db
# https://chatgpt.com/share/69905d3a-cb80-8013-a0be-5b7b799c5954
# https://chatgpt.com/share/69905d90-3474-8013-8db7-f87415685c7e
# ============================================================
import hashlib
import json
import os
import sqlite3
import re
import streamlit as st
from datetime import datetime, timezone

from langchain_core.messages import messages_from_dict, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import SQLChatMessageHistory

from chains.rag_chain import build_rag_chain
from datasets.questionnaire import QuestionnaireLogger
from datasets.telemetry import TelemetryLogger
from inputs import (DB_PATH, TELEMETRY_DB_PATH, vd_path, collection_name, DEFAULT_SLIDERS, DEFAULT_RAG_SYSTEM, \
    DEFAULT_RAG_HUMAN, DEFAULT_REPHRASE_HUMAN, DEFAULT_REPHRASE_SYSTEM, generated_rag_system_prompt, \
    QUESTIONNAIRE_DB_PATH, CLIENT_PROFILE_TEMPLATE, GUARDRAIL_WINDOW_NOTE, GUARDRAIL_LOCK_AFTER,
                    GUARDRAIL_LOCK_MINUTES, PRE_SURVEY, POST_SURVEY)
from llm.models import load_llms

# -----------------------------
# Admin auth (simple)
# -----------------------------
def ensure_state_defaults():
    st.session_state.setdefault("is_admin", False)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", "user_3")
    st.session_state.setdefault("last_session_id", st.session_state.session_id)

    # Default prompt texts stored in session so admin_pages can edit
    st.session_state.setdefault("rag_system_prompt", DEFAULT_RAG_SYSTEM)
    st.session_state.setdefault("rag_human_template", DEFAULT_RAG_HUMAN)

    st.session_state.setdefault("rephrase_system_prompt", DEFAULT_REPHRASE_SYSTEM)
    st.session_state.setdefault("rephrase_human_template", DEFAULT_REPHRASE_HUMAN)

    # Default strategy sliders (same defaults as admin_pages/prompts.py)
    st.session_state.setdefault("price_policy", DEFAULT_SLIDERS["price_policy"])
    st.session_state.setdefault("response_format", DEFAULT_SLIDERS["response_format"])
    st.session_state.setdefault("response_length", DEFAULT_SLIDERS["response_length"])

    st.session_state.setdefault("user_settings_version", 0)
    st.session_state.setdefault("chain_built_for_settings_version", -1)

    # Chain rebuild toggle key
    st.session_state.setdefault("chain_version", 0)

    st.session_state.setdefault("pre_survey_done", False)
    st.session_state.setdefault("post_survey_done", False)
    st.session_state.setdefault("show_post_survey", False)

    st.session_state.setdefault("client_profile", "")

    st.session_state.setdefault("guardrail_hits", 0)
    st.session_state.setdefault("locked_until_utc", None)


def admin_login_ui():
    if st.session_state.is_admin:
        st.success("Admin mode enabled.")
        if st.button("Exit admin_pages mode"):
            st.session_state.is_admin = False
            st.rerun()
        return

    with st.expander("Admin mode", expanded=False):
        admin_pass = st.text_input("Admin password", type="password")
        if st.button("Enter admin_pages mode"):
            expected = st.secrets.get("ADMIN_PASSWORD")
            if expected is None:
                st.error("ADMIN_PASSWORD not configured in secrets.toml")
                return

            if admin_pass == expected:
                st.session_state.is_admin = True
                st.success("Admin mode enabled.")
                st.rerun()
            else:
                st.error("Wrong password.")



def load_survey_status_for_session(session_id: str):
    quest = QuestionnaireLogger(QUESTIONNAIRE_DB_PATH)
    s = quest.get_user_survey(session_id) or {}
    st.session_state.pre_survey_done = bool(s.get("pre"))
    st.session_state.post_survey_done = bool(s.get("post"))
    if st.session_state.post_survey_done:
        st.session_state.show_post_survey = False



def effective_sliders_from_state() -> dict:
    return {
        "price_policy": st.session_state.get("price_policy") or DEFAULT_SLIDERS["price_policy"],
        "response_format": st.session_state.get("response_format") or DEFAULT_SLIDERS["response_format"],
        "response_length": st.session_state.get("response_length") or DEFAULT_SLIDERS["response_length"],
        "settings_version": int(st.session_state.get("user_settings_version", 0)),
    }


def load_sliders_for_session(session_id: str):
    tel = TelemetryLogger(TELEMETRY_DB_PATH)
    saved = tel.get_user_sliders(session_id)
    sliders = {**DEFAULT_SLIDERS, **(saved or {})}

    st.session_state.price_policy = sliders["price_policy"]
    st.session_state.response_format = sliders["response_format"]
    st.session_state.response_length = sliders["response_length"]

    st.session_state.rag_system_prompt = generated_rag_system_prompt(st.session_state.price_policy,
                                                                     st.session_state.response_format,
                                                                     st.session_state.response_length)

    st.session_state.user_settings_version = int(sliders.get("settings_version", 0))
    # If no saved settings existed yet, persist defaults now
    if saved is None:
        tel.set_user_sliders([session_id], sliders)


def get_db_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=f"sqlite:///{DB_PATH}"
    )


def load_telemetry_images_by_session(session_id: str):
    """
    returns list of per-turn image abs paths in chronological order of turns.
    Each turn corresponds to one assistant message (best-effort).
    """
    if not os.path.exists(TELEMETRY_DB_PATH):
        return []

    con = sqlite3.connect(TELEMETRY_DB_PATH)
    cur = con.cursor()
    cur.execute(
        "SELECT id, images_json FROM turns WHERE session_id=? ORDER BY id ASC",
        (session_id,)
    )
    rows = cur.fetchall()
    con.close()

    out = []
    for _, images_json in rows:
        try:
            arr = json.loads(images_json or "[]")
        except Exception:
            arr = []
        abs_paths = []
        for im in arr:
            rel = im.get("rel_path")
        out.append(abs_paths)
    return out


def load_ui_messages_from_db(session_id: str):
    hist = get_db_history(session_id)
    ui = []
    for m in hist.messages:
        role = "assistant" if m.type in ("ai", "assistant") else "user"
        item = {"role": role, "content": m.content}
        # pull images + run_id from additional_kwargs if present
        akw = getattr(m, "additional_kwargs", {}) or {}
        if role == "assistant":
            if isinstance(akw.get("images_abs"), list):
                item["images"] = akw["images_abs"]
            if akw.get("run_id"):
                item["run_id"] = akw["run_id"]
        ui.append(item)
    st.session_state.messages = ui

def render_survey_form(form_key: str, title: str, caption: str, questions: list[dict], submit_label: str):
    """
    Renders a survey from a list of question specs and returns (submitted: bool, answers: dict).
    Only edits needed for new questions: update inputs.py question list.
    """
    st.subheader(title)
    st.caption(caption)

    answers = {}

    with st.form(form_key, clear_on_submit=False):
        for q in questions:
            qid = q["id"]
            qtype = q["type"]
            label = q.get("label", qid)
            required = bool(q.get("required", False))

            # Use stable Streamlit keys so values persist while editing
            skey = f"{form_key}__{qid}"

            if qtype == "text":
                answers[qid] = st.text_input(label, value=q.get("default", ""), key=skey)

            elif qtype == "textarea":
                answers[qid] = st.text_area(label, value=q.get("default", ""), key=skey)

            elif qtype == "date":
                answers[qid] = st.date_input(label, key=skey)

            elif qtype == "select":
                options = q["options"]
                default = q.get("default", options[0])
                try:
                    idx = options.index(default)
                except ValueError:
                    idx = 0
                answers[qid] = st.selectbox(label, options, index=idx, key=skey)

            elif qtype == "radio":
                options = q["options"]
                default = q.get("default", options[0])
                try:
                    idx = options.index(default)
                except ValueError:
                    idx = 0
                answers[qid] = st.radio(
                    label,
                    options,
                    index=idx,
                    horizontal=bool(q.get("horizontal", False)),
                    key=skey,
                )

            elif qtype == "slider":
                mn = int(q["min"])
                mx = int(q["max"])
                default = int(q.get("default", mn))
                answers[qid] = st.slider(label, mn, mx, default, key=skey)

            elif qtype == "checkbox":
                default = bool(q.get("default", False))
                answers[qid] = st.checkbox(label, value=default, key=skey)

            elif qtype == "number":
                default = q.get("default", 0)
                answers[qid] = st.number_input(label, value=default, key=skey)

            else:
                st.error(f"Unknown question type: {qtype}")
                st.stop()

        submitted = st.form_submit_button(submit_label)

        if submitted:
            # Validation: required fields
            missing = []
            for q in questions:
                if not q.get("required"):
                    continue
                qid = q["id"]
                val = answers.get(qid)

                # Required checkbox must be True
                if q["type"] == "checkbox":
                    if val is not True:
                        missing.append(q.get("label", qid))
                    continue

                # Required text/textarea must be non-empty
                if q["type"] in ("text", "textarea"):
                    if not str(val).strip():
                        missing.append(q.get("label", qid))
                    continue

                # For other types, just ensure not None
                if val is None:
                    missing.append(q.get("label", qid))

            if missing:
                st.error("Please complete required fields:\n- " + "\n- ".join(missing))
                return False, answers

            return True, answers

    return False, answers


def on_session_change(new_session_id: str):
    st.session_state["session_id"] = new_session_id
    st.session_state["last_session_id"] = new_session_id

    load_ui_messages_from_db(new_session_id)
    load_sliders_for_session(new_session_id)
    load_survey_status_for_session(new_session_id)
    load_moderation_status_for_session(st.session_state["session_id"])

    st.session_state["show_post_survey"] = False

    load_or_build_client_profile(new_session_id)


def render_pre_survey_if_needed():
    if st.session_state.get("pre_survey_done"):
        return

    sid = st.session_state.get("session_id")
    if not sid:
        st.error("No user id selected yet.")
        st.stop()

    submitted, answers = render_survey_form(
        form_key="pre_survey",
        title="Before you start",
        caption="Please answer a few questions before using the chatbot.",
        questions=PRE_SURVEY,
        submit_label="Start chat",
    )

    if submitted:
        quest = QuestionnaireLogger(QUESTIONNAIRE_DB_PATH)
        # store normalized values (dates -> str)
        stored = {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in answers.items()}
        quest.upsert_pre_survey(sid, stored)
        st.session_state["pre_survey_done"] = True

        # Build and store default profile from this pre-survey
        profile_text = build_profile_from_pre(stored)
        quest.upsert_user_profile(sid, profile_text, source="auto")
        st.session_state["client_profile"] = profile_text

        st.rerun()

def render_post_survey_if_needed():
    if st.session_state.get("post_survey_done"):
        st.success("Thanks! Your post-chat questionnaire has been recorded.")
        return
    if not st.session_state.get("show_post_survey"):
        return

    sid = st.session_state.get("session_id")
    if not sid:
        st.error("No user id selected yet.")
        st.stop()

    submitted, answers = render_survey_form(
        form_key="post_survey",
        title="After chatting",
        caption="A few quick questions to wrap up your session.",
        questions=POST_SURVEY,
        submit_label="Submit",
    )

    if submitted:
        quest = QuestionnaireLogger(QUESTIONNAIRE_DB_PATH)
        stored = {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in answers.items()}
        quest.upsert_post_survey(sid, stored)
        st.session_state["post_survey_done"] = True
        st.session_state["show_post_survey"] = False
        st.rerun()

def build_profile_from_pre(pre: dict) -> str:
    pre = pre or {}
    def g(k, default=""):
        v = pre.get(k, default)
        return str(v) if v is not None else ""

    return CLIENT_PROFILE_TEMPLATE.format(
        profession=g("profession", "Unknown"),
        education=g("education", "Unknown"),
        ai_knowledge=g("ai_knowledge", "Unknown"),
        journey_stage=g("journey_stage", "Unknown"),
        purchase_timeframe=g("purchase_timeframe", "Unknown"),
        budget_range=g("budget_range", "Unknown"),
        ev_experience=g("ev_experience", "Unknown"),
        home_charging=g("home_charging", "Unknown"),
        trust_ai=g("trust_ai", ""),
    )

def load_or_build_client_profile(session_id: str):
    quest = QuestionnaireLogger(QUESTIONNAIRE_DB_PATH)

    # 1) If a manual/previous profile exists -> load it
    existing = quest.get_user_profile(session_id)
    if existing:
        st.session_state["client_profile"] = existing
        return

    # 2) Otherwise, if pre-survey exists -> build profile from it, store as "auto"
    s = quest.get_user_survey(session_id) or {}
    pre = s.get("pre")
    if pre:
        prof = build_profile_from_pre(pre)
        quest.upsert_user_profile(session_id, prof, source="auto")
        st.session_state["client_profile"] = prof
        return

    # 3) Otherwise nothing yet (pre-survey will fill later)
    st.session_state["client_profile"] = ""

# --- Guardrail settings ---
def utc_now():
    return datetime.now(timezone.utc)
def _parse_iso(s: str | None):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def load_moderation_status_for_session(session_id: str):
    tel = TelemetryLogger(TELEMETRY_DB_PATH)
    stt = tel.get_user_moderation(session_id)
    st.session_state["guardrail_hits"] = int(stt.get("guardrail_hits") or 0)
    st.session_state["locked_until_utc"] = stt.get("locked_until_utc")

def is_locked_now() -> tuple[bool, str | None]:
    locked_until = _parse_iso(st.session_state.get("locked_until_utc"))
    if locked_until and locked_until > utc_now():
        return True, locked_until.isoformat()
    return False, None

GUARDRAIL_PATTERN = re.compile(r"(?mi)^\s*GUARDRAIL:\s*([a-z_]+)\s*$")

COUNTED_GUARDRAILS = {"off_topic", "prompt_injection", "privacy", "unsafe"}  # do NOT count "no_context"

def extract_guardrail(answer_text: str) -> tuple[str | None, str]:
    """
    Returns (guardrail_type, cleaned_answer_without_tag)
    """
    m = GUARDRAIL_PATTERN.search(answer_text or "")
    if not m:
        return None, answer_text

    gtype = m.group(1).strip().lower()
    # remove the guardrail line from the answer the user sees
    cleaned = GUARDRAIL_PATTERN.sub("", answer_text).strip()
    return gtype, cleaned




# -----------------------------
# Prompt builders (from state)
# -----------------------------
def build_prompt_rag():
    return ChatPromptTemplate.from_messages([
        ("system", st.session_state.rag_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", st.session_state.rag_human_template),
    ])


def build_prompt_rephrase():
    return ChatPromptTemplate.from_messages([
        ("system", st.session_state.rephrase_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", st.session_state.rephrase_human_template),
    ])

def prompt_fingerprint() -> str:
    s = "\n---\n".join([
        st.session_state.get("rag_system_prompt", ""),
        st.session_state.get("rag_human_template", ""),
        st.session_state.get("rephrase_system_prompt", ""),
        st.session_state.get("rephrase_human_template", ""),
        # sliders too, because they should reflect in prompt generation
        st.session_state.get("price_policy", ""),
        st.session_state.get("response_format", ""),
        st.session_state.get("response_length", ""),
        str(st.session_state.get("user_settings_version", 0)),
    ])
    return hashlib.sha256(s.encode("utf-8")).hexdigest()



# -----------------------------
# Chain init (rebuildable)
# -----------------------------
@st.cache_resource(show_spinner=True)
def init_models():
    llm_or, _ = load_llms()
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    return llm_or, embeddings


def build_chain_cached(chain_version: int, prompt_fp: str, model_fp: str):
    llm_or, embeddings = init_models()

    trimmer = trim_messages(
        max_tokens=10,
        strategy="last",
        token_counter=len,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    prompt_rag = build_prompt_rag()
    prompt_rephrase = build_prompt_rephrase()

    chain, retriever_hist = build_rag_chain(
        llm=llm_or,
        embeddings=embeddings,
        chroma_path=vd_path,
        collection_name=collection_name,
        prompt_rag=prompt_rag,
        trimmer=trimmer,
        prompt_rephrase=prompt_rephrase,
    )

    return chain, retriever_hist


@st.cache_resource(show_spinner=True)
def get_chain(chain_version: int, prompt_fp: str, model_fp: str):
    return build_chain_cached(chain_version, prompt_fp, model_fp)


# -----------------------------
# Streamlit App
# -----------------------------
ensure_state_defaults()
if st.session_state.get("last_session_id") != st.session_state.get("session_id"):
    on_session_change(st.session_state["session_id"])
else:
    # first run: ensure we loaded status at least once
    load_sliders_for_session(st.session_state["session_id"])
    load_survey_status_for_session(st.session_state["session_id"])
    load_moderation_status_for_session(st.session_state["session_id"])
    if not st.session_state.get("messages"):
        load_ui_messages_from_db(st.session_state["session_id"])
if st.session_state.chain_built_for_settings_version != st.session_state.user_settings_version:
    st.session_state.chain_version += 1
    st.session_state.chain_built_for_settings_version = st.session_state.user_settings_version

st.title("🚗 Car Search Journey Assistant")

# Sidebar (always visible)
with st.sidebar:
    if not st.session_state.messages:
        load_ui_messages_from_db(st.session_state.session_id)

    st.header("Session")
    st.session_state.session_id = st.text_input("session_id", value=st.session_state.session_id)

    # ✅ If session changes, reload transcript from DB
    if st.session_state.session_id != st.session_state.last_session_id:
        st.session_state.last_session_id = st.session_state.session_id
        load_ui_messages_from_db(st.session_state.session_id)
        load_sliders_for_session(st.session_state.session_id)
        load_survey_status_for_session(st.session_state.session_id)
        load_moderation_status_for_session(st.session_state["session_id"])
        st.rerun()

    st.header("Client context")

    # Make sure we have something loaded
    if not st.session_state.get("client_profile"):
        load_or_build_client_profile(st.session_state.session_id)

    edited = st.text_area(
        "Client profile",
        value=st.session_state.get("client_profile", ""),
        height=350,
        key="client_profile_editor",
    )

    col1, col2 = st.columns([2,3])
    with col1:
        if st.button("Save profile"):
            quest = QuestionnaireLogger(QUESTIONNAIRE_DB_PATH)
            quest.upsert_user_profile(st.session_state.session_id, edited, source="manual")
            st.session_state["client_profile"] = edited
            st.success("Profile saved.")
    with col2:
        if st.button("Reset from survey"):
            quest = QuestionnaireLogger(QUESTIONNAIRE_DB_PATH)
            s = quest.get_user_survey(st.session_state.session_id) or {}
            pre = s.get("pre") or {}
            rebuilt = build_profile_from_pre(pre)
            quest.upsert_user_profile(st.session_state.session_id, rebuilt, source="auto")
            st.session_state["client_profile"] = rebuilt
            st.success("Profile rebuilt from pre-survey.")

    if st.button("Clear UI chat"):
        st.session_state.messages = []

    # Admin gate
    admin_login_ui()

render_pre_survey_if_needed()
if not st.session_state.get("pre_survey_done"):
    st.stop()
render_post_survey_if_needed()
if st.session_state.get("show_post_survey") and not st.session_state.get("post_survey_done"):
    st.stop()

# Build / get chain (depends on current version)
def _model_fingerprint(llm) -> str:
    return str(getattr(llm, "model", None) or getattr(llm, "model_name", None) or "")

# Build / get chain
llm_or, embeddings = init_models()
model_fp = _model_fingerprint(llm_or)

prompt_fp = prompt_fingerprint()
chain, retriever_hist = get_chain(
    st.session_state.chain_version,
    prompt_fp,
    model_fp,   # NEW
)

# Render chat transcript
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg.get("content", ""))

        # ✅ re-render stored images
        imgs = msg.get("images") or []
        if imgs:
            st.divider()
            st.caption("Images")
            for img_path in imgs:
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.caption(f"Missing file: {img_path}")

# Chat input + response
if not st.session_state.get("post_survey_done"):
    if st.button("End conversation"):
        st.session_state.show_post_survey = True
        st.rerun()

locked, until_iso = is_locked_now()
if locked:
    st.error(f"🚫 This user session is temporarily locked until {until_iso} (UTC). {GUARDRAIL_WINDOW_NOTE}")
    st.stop()

user_question = st.chat_input("Type your question…")
if user_question:
    load_sliders_for_session(st.session_state.session_id)
    # then recompute prompt_fp and get_chain AFTER that:
    prompt_fp = prompt_fingerprint()
    chain, retriever_hist = get_chain(st.session_state.chain_version, prompt_fp, model_fp)
    # 1) load DB history
    hist = get_db_history(st.session_state.session_id)

    # 2) write user msg into DB
    hist.add_message(HumanMessage(content=user_question))

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Build chain input
            chain_input = {
                "question": user_question,
                "client_context": st.session_state.get("client_profile", ""),
                "chat_history": hist.messages,
                "session_id": st.session_state.session_id,

                "sliders": effective_sliders_from_state(),

                "prompt_templates": {
                    "rag_system": "streamlit_state",
                    "rephrase_system": "streamlit_state",
                },
            }
            response = chain.invoke(chain_input)
            st.sidebar.write(response)
            assistant_text_raw = response.get("answer", str(response))
            images_abs = response.get("images_abs", []) or []
            run_id = response.get("run_id")

            gtype, assistant_text = extract_guardrail(assistant_text_raw)  # assistant_text now cleaned

            st.markdown(assistant_text)

            if gtype in COUNTED_GUARDRAILS:
                tel = TelemetryLogger(TELEMETRY_DB_PATH)
                new_state = tel.record_guardrail_hit(
                    session_id=st.session_state.session_id,
                    trigger_type=gtype,
                    question_excerpt=user_question,
                    lock_after=GUARDRAIL_LOCK_AFTER,
                    lock_minutes=GUARDRAIL_LOCK_MINUTES,
                )
                st.session_state["guardrail_hits"] = int(new_state.get("guardrail_hits") or 0)
                st.session_state["locked_until_utc"] = new_state.get("locked_until_utc")
            locked2, until_iso2 = is_locked_now()
            if locked2:
                st.error(f"🚫 This user session is now locked until {until_iso2} (UTC). {GUARDRAIL_WINDOW_NOTE}")
                st.stop()

            if images_abs:
                st.divider()
                st.caption("Images")
                for p in images_abs:
                    if os.path.exists(p):
                        st.image(p, use_container_width=True)
                    else:
                        st.caption(f"Missing file: {p}")

    # 3) write assistant msg into DB with metadata
    hist.add_message(AIMessage(
        content=assistant_text,
        additional_kwargs={
            "run_id": run_id,
            "images_abs": images_abs,
        }
    ))

    # 4) reload UI messages from DB
    load_ui_messages_from_db(st.session_state.session_id)
    st.rerun()
