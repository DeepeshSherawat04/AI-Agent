import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
Conversation memory management for GigaCorp Support Agent.
Uses LangChain's ConversationBufferWindowMemory for proper memory management.
"""

from typing import List, Tuple

from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage

class ConversationMemoryManager:
    """
    Manages conversational memory with a sliding window using LangChain.
    Keeps the last K exchanges to maintain context without exceeding token limits.
    """
    
    def __init__(self, k: int = 5):
        self.k = k
        self.memories: dict = {}  # session_id -> ConversationBufferWindowMemory
        
    def get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Get or create LangChain memory for a session."""
        if session_id not in self.memories:
            self.memories[session_id] = ConversationBufferWindowMemory(
                k=self.k,
                return_messages=True,
                memory_key="chat_history",
                input_key="question",
                output_key="answer"
            )
        return self.memories[session_id]
    
    def add_exchange(self, session_id: str, question: str, answer: str):
        """Add a question-answer exchange to memory."""
        memory = self.get_memory(session_id)
        memory.save_context(
            {"question": question},
            {"answer": answer}
        )
    
    def get_chat_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Get formatted chat history for a session."""
        if session_id not in self.memories:
            return []
        
        memory = self.memories[session_id]
        messages = memory.load_memory_variables({}).get("chat_history", [])
        
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append(("user", msg.content))
            elif isinstance(msg, AIMessage):
                history.append(("agent", msg.content))
        
        return history
    
    def clear_memory(self, session_id: str):
        """Clear memory for a specific session."""
        if session_id in self.memories:
            del self.memories[session_id]
    
    def get_memory_context(self, session_id: str) -> str:
        """Get memory as a formatted string for prompt injection."""
        if session_id not in self.memories:
            return ""
        
        memory = self.memories[session_id]
        variables = memory.load_memory_variables({})
        return variables.get("chat_history", "")


# Global memory manager instance
memory_manager = ConversationMemoryManager()