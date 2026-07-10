"""
GigaCorp Customer Support Agent - Streamlit Web Application
A RAG-powered support chatbot with memory and source citations.
"""

import streamlit as st
import uuid
from src.agents.support_agent import get_agent

# Page configuration
st.set_page_config(
    page_title="GigaCorp Support Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
    }
    .agent-message {
        background-color: #F5F5F5;
        border-left: 4px solid #43A047;
    }
    .source-box {
        background-color: #FFF8E1;
        border: 1px solid #FFB300;
        border-radius: 0.3rem;
        padding: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .sidebar-info {
        font-size: 0.9rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_agent_instance():
    """Initialize agent once and cache it across reruns."""
    return get_agent()


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    
    if "agent" not in st.session_state:
        with st.spinner("🚀 Initializing AI Agent... (One-time setup: downloading AI models ~30s)"):
            st.session_state.agent = get_agent_instance()
            st.session_state.session_id = st.session_state.agent.chat("Hello")["session_id"]


def display_chat_history():
    """Display all messages in the chat history."""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>{msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            sources_html = ""
            if msg.get("sources"):
                sources_html = '<div class="source-box"><strong>📚 Sources:</strong><ul>'
                for src in msg["sources"]:
                    sources_html += f'<li>{src["source"]} (relevance: {src["relevance_score"]})</li>'
                sources_html += '</ul></div>'
            
            st.markdown(f"""
            <div class="chat-message agent-message">
                <strong>🤖 GigaCorp Agent:</strong><br>{msg["content"]}
                {sources_html}
            </div>
            """, unsafe_allow_html=True)


def main():
    # Sidebar
    with st.sidebar:
        st.markdown('<p class="main-header">🤖 GigaCorp Support</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-Powered Customer Support</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        <div class="sidebar-info">
        This AI agent uses <b>Retrieval-Augmented Generation (RAG)</b> to answer your questions 
        accurately using GigaCorp's knowledge base.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🧠 Features")
        st.markdown("""
        - ✅ Context-aware conversations
        - ✅ Source citations
        - ✅ Memory across chat turns
        - ✅ Local embeddings (privacy-first)
        """)
        
        st.markdown("### 📋 Sample Questions")
        st.markdown("""
        - "Do you ship to India?"
        - "How much does shipping cost?"
        - "What is your return policy?"
        - "What are your business hours?"
        - "Tell me about Pro plan benefits"
        """)
        
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            if st.session_state.session_id:
                st.session_state.agent.clear_session(st.session_state.session_id)
                st.session_state.session_id = None
            st.rerun()
        
        st.markdown("---")
        
        # Agent stats
        if "agent" in st.session_state:
            stats = st.session_state.agent.get_stats()
            st.markdown("### 📊 Agent Stats")
            st.markdown(f"""
            - **Model:** `{stats['llm_model']}`
            - **Embeddings:** `{stats['embedding_model']}`
            - **Active Sessions:** {stats['active_sessions']}
            """)
    
    # Main content
    st.markdown('<p class="main-header">🤖 GigaCorp Support Agent</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask me anything about shipping, returns, service plans, and more!</p>', unsafe_allow_html=True)
    
    # Initialize
    init_session_state()
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Get agent response
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.agent.chat(
                message=prompt,
                session_id=st.session_state.session_id
            )
            
            # Update session ID if new
            if st.session_state.session_id is None:
                st.session_state.session_id = response["session_id"]
        
        # Add agent message
        st.session_state.messages.append({
            "role": "agent",
            "content": response["answer"],
            "sources": response.get("sources", []),
            "retrieved_count": response.get("retrieved_count", 0)
        })
        
        # Rerun to display new messages
        st.rerun()


if __name__ == "__main__":
    main()