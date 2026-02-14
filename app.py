# ============================================================
# This code is created with the help of AI. As a starting
# point, there was already a working code where a RAG chain
# could be invoked. ChatGPT 5.2 was used to craft a streamlit
# application around this code. The chats can be found at:
# https://chatgpt.com/share/6990586a-0d68-8013-880f-243ee001d006
# https://chatgpt.com/share/699059a9-13a0-8013-ad55-189696111186
# ============================================================
import streamlit as st

st.set_page_config(page_title="EV Chat", page_icon="🚗", layout="centered")

# Ensure state
st.session_state.setdefault("is_admin", False)

# Define pages
home_page = st.Page("home.py", title="Home", icon="🏠")

admin_pages = [
    st.Page("admin_pages/chat_database.py", title="Chats database", icon="🗄️"),
    st.Page("admin_pages/questionnaire_database.py", title="Questionnaire database", icon="🧾"),
    st.Page("admin_pages/prompts.py", title="Prompts", icon="🧠"),
]


# Only include admin_pages pages if admin_pages
pages = [home_page]
if st.session_state.is_admin:
    pages += admin_pages

pg = st.navigation(pages)
pg.run()
