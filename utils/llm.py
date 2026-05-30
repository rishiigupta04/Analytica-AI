# =============================================================
# utils/llm.py — Shared LLM Initializer
# =============================================================
# One place to configure the language model.
# All 4 agents import from here — so if you ever want to
# swap Groq for Gemini or change the model, you change it
# in ONE place and every agent automatically uses it.
# =============================================================

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # Reads your .env file and loads GROQ_API_KEY into the environment


def get_llm(temperature: float = 0) -> ChatGroq:
    """
    Returns a configured Groq LLM instance.

    Args:
        temperature: Controls creativity. 0 = deterministic/factual (best for data analysis).
                     Higher values (0.7-1.0) = more creative (better for writing).

    Returns:
        ChatGroq: A ready-to-use LLM object.

    Raises:
        ValueError: If GROQ_API_KEY is missing from .env
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "\n\n❌ GROQ_API_KEY not found or not set!\n"
        )

    return ChatGroq(
        model="llama-3.3-70b-versatile",  # Best free model on Groq as of 2025
        temperature=temperature,
        api_key=api_key,
        max_retries=2,  # Auto-retry on transient API failures
    )