"""
GigaCorp Support Agent - Main orchestrator.
Combines RAG retrieval, conversation memory, and Groq LLM for intelligent support.
"""

import uuid
from typing import Dict, Any, List, Tuple

from src.chains.rag_chain import RAGChain
from src.memory.conversation_memory import memory_manager
from src.retrieval.vector_store import initialize_vector_store, GigaCorpVectorStore


class GigaCorpSupportAgent:
    """
    Main customer support agent that handles user queries with:
    - RAG-based knowledge retrieval
    - Conversational memory across turns
    - Source citations
    - Context-aware follow-up handling
    """
    
    def __init__(self):
        print("🤖 Initializing GigaCorp Support Agent...")
        self.vector_store = initialize_vector_store()
        self.rag_chain = RAGChain(self.vector_store)
        print("✅ Agent ready!")
        
    def chat(self, message: str, session_id: str = None) -> Dict[str, Any]:
        """
        Process a user message and return a response with sources.
        
        Args:
            message: User's question
            session_id: Optional session ID for memory persistence
            
        Returns:
            Dict containing answer, sources, and metadata
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Get conversation history for context
        chat_history = memory_manager.get_memory_context(session_id)
        
        # Run RAG chain
        result = self.rag_chain.run(
            question=message,
            chat_history=chat_history
        )
        
        # Store exchange in memory
        memory_manager.add_exchange(session_id, message, result["answer"])
        
        # Add session info to result
        result["session_id"] = session_id
        result["chat_history"] = memory_manager.get_chat_history(session_id)
        
        return result
    
    def get_session_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Get full chat history for a session."""
        return memory_manager.get_chat_history(session_id)
    
    def clear_session(self, session_id: str):
        """Clear a specific conversation session."""
        memory_manager.clear_memory(session_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "active_sessions": len(memory_manager.memories),
            "vector_store_path": str(self.vector_store.vector_store_path),
            "llm_model": self.rag_chain.model,
            "embedding_model": self.vector_store.embeddings.model_name
        }


# Global agent instance (singleton)
_support_agent: GigaCorpSupportAgent = None

def get_agent() -> GigaCorpSupportAgent:
    """Get or create the global agent instance."""
    global _support_agent
    if _support_agent is None:
        _support_agent = GigaCorpSupportAgent()
    return _support_agent