import os
import sys

# ── Streamlit config via env (before streamlit imports) ─────────────────────
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# ── Python version guard ────────────────────────────────────────────────────
_PY_VER = sys.version_info
if _PY_VER < (3, 10):
    raise RuntimeError(
        f"Python {_PY_VER.major}.{_PY_VER.minor} is not supported. "
        "Please use Python 3.10 or higher."
    )
if _PY_VER >= (3, 13):
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater",
        category=UserWarning,
    )

# ── Environment & HF auth ───────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

# ── App imports ─────────────────────────────────────────────────────────────
import html as html_lib
import streamlit as st
from src.agents.support_agent import get_agent

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="GigaCorp Support Agent",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="expanded",
)

QUICK_QUESTIONS = [
    "Do you ship to India?",
    "How much does shipping cost?",
    "What is your return policy?",
    "What are your business hours?",
    "Tell me about Pro plan benefits",
]

# ----------------------------------------------------------------------------
# Styling — "support console" identity: navy sidebar, teal (agent) / indigo
# (you) message accents, and an amber "ledger" treatment for citations.
# ----------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --navy: #111827;
        --cloud: #F6F7FB;
        --teal: #0F766E;
        --teal-tint: #F0FDFA;
        --indigo: #4F46E5;
        --indigo-tint: #EEF2FF;
        --amber: #F59E0B;
        --amber-tint: #FFFBEB;
        --text: #111827;
        --muted: #6B7280;
        --border: #E5E7EB;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: var(--cloud); }
    .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 880px; }
    footer { visibility: hidden; }

    /* ---------- Header ---------- */
    .app-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        color: var(--navy);
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.1rem;
    }
    .status-dot {
        width: 9px; height: 9px; border-radius: 50%;
        background: #22C55E;
        box-shadow: 0 0 0 3px rgba(34,197,94,0.18);
        display: inline-block;
    }
    .app-subtitle { color: var(--muted); font-size: 0.98rem; margin-bottom: 0.6rem; }
    .header-rule {
        height: 3px; width: 64px; border-radius: 2px;
        background: linear-gradient(90deg, var(--teal), var(--amber));
        margin: 0.4rem 0 1.8rem 0;
    }

    /* ---------- Chat bubbles ---------- */
    .msg-row { display: flex; align-items: flex-start; gap: 0.6rem; margin-bottom: 1.15rem; animation: fadeIn 0.25s ease; }
    .user-row { justify-content: flex-end; }
    .agent-row { justify-content: flex-start; }
    .avatar {
        width: 34px; height: 34px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.05rem; flex-shrink: 0;
    }
    .user-avatar { background: var(--indigo-tint); }
    .agent-avatar { background: var(--teal-tint); }
    .bubble {
        max-width: 75%; padding: 0.85rem 1.05rem; border-radius: 1rem;
        font-size: 0.96rem; line-height: 1.55; color: var(--text);
    }
    .user-bubble {
        background: var(--indigo-tint); border: 1px solid #DCE0FA;
        border-top-right-radius: 0.3rem;
    }
    .agent-bubble {
        background: #fff; border: 1px solid var(--border);
        border-top-left-radius: 0.3rem;
        box-shadow: 0 1px 2px rgba(17,24,39,0.05);
    }

    /* ---------- Citations — styled like ticket stubs ---------- */
    .sources-wrap { margin-top: 0.75rem; border-top: 1px dashed var(--border); padding-top: 0.6rem; }
    .sources-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
        letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted);
        margin-bottom: 0.45rem;
    }
    .chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; }
    .citation-chip {
        display: flex; align-items: center; gap: 0.35rem;
        background: var(--amber-tint); border: 1px dashed #F3C77A;
        border-radius: 0.5rem; padding: 0.3rem 0.55rem;
        font-size: 0.78rem; color: #92620A;
    }
    .chip-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--amber); flex-shrink: 0; }
    .chip-score { font-family: 'JetBrains Mono', monospace; margin-left: 0.15rem; opacity: 0.85; }

    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

    /* ---------- Sidebar — dark console panel ---------- */
    section[data-testid="stSidebar"] { background: var(--navy); }
    section[data-testid="stSidebar"] * { color: #E5E7EB; }
    section[data-testid="stSidebar"] h3 {
        font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem;
        text-transform: uppercase; letter-spacing: 0.05em; color: #9CA3AF !important;
        margin-top: 1.5rem; margin-bottom: 0.6rem;
        border-top: 1px solid rgba(255,255,255,0.08); padding-top: 1.2rem;
    }
    .brand-title {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.3rem;
        color: #fff; display: flex; align-items: center; gap: 0.4rem;
    }
    .brand-sub { color: #9CA3AF; font-size: 0.85rem; margin-top: 0.15rem; }
    section[data-testid="stSidebar"] .stButton>button {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.14);
        color: #E5E7EB; text-align: left; border-radius: 0.55rem;
        font-size: 0.83rem; padding: 0.5rem 0.8rem; width: 100%;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        border-color: var(--teal); color: #fff;
        background: rgba(15,118,110,0.3);
    }
    .stats-card {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 0.7rem; padding: 0.9rem 1rem;
        font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;
        line-height: 1.8; color: #D1D5DB;
    }
    .stats-card b { color: #fff; }

    /* ---------- Chat input ---------- */
    .stChatInput textarea { font-family: 'Inter', sans-serif; }
    </style>
    """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Agent setup
# ----------------------------------------------------------------------------
@st.cache_resource
def get_agent_instance():
    """Initialize agent once and cache it across reruns."""
    return get_agent()


def init_session_state():
    """Initialize Streamlit session state variables with persistent session_id."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # CRITICAL: session_id persists across reruns and refreshes
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    if "agent" not in st.session_state:
        with st.spinner("🎧 Waking up the support agent — first run downloads models (~30s)..."):
            try:
                st.session_state.agent = get_agent_instance()
                # Only generate session_id once on first load
                if st.session_state.session_id is None:
                    greeting = st.session_state.agent.chat("Hello")
                    st.session_state.session_id = greeting["session_id"]
            except Exception as exc:
                st.error(f"Couldn't start the agent: {exc}")
                st.stop()


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def render_user_message(content: str):
    safe = html_lib.escape(content).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="msg-row user-row">
            <div class="bubble user-bubble">{safe}</div>
            <div class="avatar user-avatar">🧑</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agent_message(content: str, sources: list):
    safe = html_lib.escape(content or "").replace("\n", "<br>")

    sources_html = ""
    if sources:
        chips = ""
        for src in sources:
            name = html_lib.escape(str(src.get("source", "Unknown source")))
            score = src.get("relevance_score")
            score_html = (
                f'<span class="chip-score">{score:.2f}</span>'
                if isinstance(score, (int, float))
                else ""
            )
            chips += (
                f'<div class="citation-chip"><span class="chip-dot"></span>'
                f"{name}{score_html}</div>"
            )
        sources_html = (
            '<div class="sources-wrap"><div class="sources-label">📎 Sources</div>'
            f'<div class="chip-row">{chips}</div></div>'
        )

    st.markdown(
        f"""
        <div class="msg-row agent-row">
            <div class="avatar agent-avatar">🎧</div>
            <div class="bubble agent-bubble">{safe}{sources_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_chat_history():
    """Display all messages in the chat history."""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            render_user_message(msg["content"])
        else:
            render_agent_message(msg["content"], msg.get("sources", []))


def handle_new_message(prompt: str):
    """Send a message to the agent and store both turns in history."""
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("🎧 Looking into that..."):
        try:
            # Always use the persisted session_id for memory continuity
            response = st.session_state.agent.chat(
                message=prompt, session_id=st.session_state.session_id
            )
        except Exception as exc:
            response = {
                "answer": f"Sorry, something went wrong on my end ({exc}). Please try again.",
                "sources": [],
                "session_id": st.session_state.session_id,
            }

    # Update session_id if agent returned a new one (safety fallback)
    if response.get("session_id"):
        st.session_state.session_id = response["session_id"]

    st.session_state.messages.append(
        {
            "role": "agent",
            "content": response.get("answer", ""),
            "sources": response.get("sources", []),
            "retrieved_count": response.get("retrieved_count", 0),
        }
    )


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="brand-title">🎧 GigaCorp Support</div>'
            '<div class="brand-sub">AI-powered customer support</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### ℹ️ About")
        st.markdown(
            "This agent uses **Retrieval-Augmented Generation (RAG)** to answer "
            "questions accurately from GigaCorp's knowledge base."
        )

        st.markdown("### 🧠 Features")
        st.markdown(
            """
            - ✅ Context-aware conversations
            - ✅ Source citations
            - ✅ Memory across chat turns
            - ✅ Local embeddings (privacy-first)
            """
        )

        st.markdown("### 💬 Try asking")
        for i, question in enumerate(QUICK_QUESTIONS):
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                st.session_state.pending_prompt = question
                st.rerun()

        st.markdown("###  ")
        if st.button("🗑️ Clear chat", type="secondary", use_container_width=True):
            st.session_state.messages = []
            # Clear agent memory but keep session alive with new greeting
            if st.session_state.get("session_id"):
                st.session_state.agent.clear_session(st.session_state.session_id)
                greeting = st.session_state.agent.chat("Hello")
                st.session_state.session_id = greeting["session_id"]
            st.rerun()

        if "agent" in st.session_state:
            stats = st.session_state.agent.get_stats()
            st.markdown("### 📊 Agent stats")
            st.markdown(
                f"""
                <div class="stats-card">
                <b>Model</b><br>{html_lib.escape(str(stats.get('llm_model', '—')))}<br><br>
                <b>Embeddings</b><br>{html_lib.escape(str(stats.get('embedding_model', '—')))}<br><br>
                <b>Active sessions</b><br>{html_lib.escape(str(stats.get('active_sessions', '—')))}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    inject_css()
    render_sidebar()

    st.markdown(
        '<div class="app-title">🎧 GigaCorp Support Agent '
        '<span class="status-dot" title="Online"></span></div>'
        '<div class="app-subtitle">Ask me anything about shipping, returns, service plans, and more.</div>'
        '<div class="header-rule"></div>',
        unsafe_allow_html=True,
    )

    init_session_state()
    display_chat_history()

    pending = st.session_state.pop("pending_prompt", None)
    typed = st.chat_input("Type your question here...")
    prompt = typed or pending

    if prompt:
        handle_new_message(prompt)
        st.rerun()


if __name__ == "__main__":
    main()