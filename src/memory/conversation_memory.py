
"""
Conversation memory management for GigaCorp Support Agent.
Modern replacement for deprecated ConversationBufferWindowMemory.
Uses only langchain_core (stable API).
"""

from typing import List, Tuple, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage


class WindowedConversationMemory:
    """
    Drop-in replacement for ConversationBufferWindowMemory.
    Keeps last k exchanges. Compatible with existing RAGChain injection.
    """
    
    def __init__(
        self,
        k: int = 5,
        return_messages: bool = True,
        memory_key: str = "chat_history",
        input_key: str = "question",
        output_key: str = "answer"
    ):
        self.k = k
        self.return_messages = return_messages
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.messages: List[HumanMessage | AIMessage] = []
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        """Add an exchange and enforce window size."""
        question = inputs.get(self.input_key, "")
        answer = outputs.get(self.output_key, "")
        
        self.messages.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])
        
        # Keep only last k exchanges (2k messages)
        if len(self.messages) > self.k * 2:
            self.messages = self.messages[-(self.k * 2):]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history in the same format as the old class."""
        if self.return_messages:
            return {self.memory_key: self.messages.copy()}
        
        # String format fallback
        lines = []
        for msg in self.messages:
            prefix = "User" if isinstance(msg, HumanMessage) else "Agent"
            lines.append(f"{prefix}: {msg.content}")
        return {self.memory_key: "\n".join(lines)}
    
    def clear(self):
        """Clear all messages."""
        self.messages = []


class ConversationMemoryManager:
    """
    Manages conversational memory with a sliding window.
    Keeps the last K exchanges to maintain context without exceeding token limits.
    """
    
    def __init__(self, k: int = 5):
        self.k = k
        self.memories: dict = {}  # session_id -> WindowedConversationMemory
        
    def get_memory(self, session_id: str) -> WindowedConversationMemory:
        """Get or create memory for a session."""
        if session_id not in self.memories:
            self.memories[session_id] = WindowedConversationMemory(
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
    
    def get_memory_context(self, session_id: str) -> Any:
        """Get memory as formatted messages/string for prompt injection."""
        if session_id not in self.memories:
            return ""
        
        memory = self.memories[session_id]
        variables = memory.load_memory_variables({})
        return variables.get("chat_history", "")


# Global memory manager instance
memory_manager = ConversationMemoryManager()