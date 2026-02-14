import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

def load_llms():
    load_dotenv(dotenv_path="C:/Users/robhe/OneDrive - Vlerick Business School/Natural Language "
                            "Processing/keys/APIkeys.py")
    llm_or = init_chat_model(
        model="mistralai/devstral-2512",
        model_provider="openai",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.5,
    )

    llm_g = init_chat_model(
        "gemma-3-27b-it",
        model_provider="google-genAI",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.5,
    )

    return llm_or, llm_g
