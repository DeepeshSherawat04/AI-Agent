"""
RAG (Retrieval-Augmented Generation) chain for GigaCorp Support Agent.
Combines retrieval from vector store with Groq LLM for accurate, cited answers.
"""

from typing import List, Dict, Any

from groq import Groq
from langchain_core.documents import Document

from src.retrieval.vector_store import GigaCorpVectorStore
from src.utils.config import settings


class RAGChain:
    """
    RAG Chain that retrieves relevant documents and generates
    accurate answers with source citations using Groq LLM.
    """

    def __init__(self, vector_store: GigaCorpVectorStore):
        self.vector_store = vector_store
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.llm_model

    def _build_prompt(self, question: str, context: str, chat_history: str = "") -> str:
        """
        Build the RAG prompt with system instructions, context, and history.
        """
        system_prompt = """You are GigaCorp's AI Customer Support Agent. Your job is to help customers by answering their questions accurately using the provided knowledge base context.

RULES:
1. Answer based on the provided context. If the answer is not in the context, say "I don't have information about that in our knowledge base. Please contact support@gigacorp.com for assistance."
2. Always cite your sources at the end of your answer using the format: [Source: <source citation>]
3. Be concise but thorough. Use bullet points for lists when appropriate.
4. If the user asks a follow-up question (like "how much does it cost?" after asking about shipping), use the chat history to understand what they're referring to.
5. Be polite, professional, and helpful.
6. For greetings like "hi", "hello", "hey" — respond warmly and briefly, then ask how you can help them today.
7. Do NOT make up information that is not in the context."""

        history_section = ""
        if chat_history:
            history_section = f"Previous conversation:\n{chat_history}\n"

        prompt = f"""{system_prompt}

{history_section}
Context from GigaCorp Knowledge Base:
{context}

---
Customer Question: {question}

Answer (with source citations, or a warm greeting if the user just said hello):"""

        return prompt

    def _format_context(self, docs: List[tuple]) -> str:
        """
        Format retrieved documents into context string with source tracking.
        """
        context_parts = []
        for i, (doc, score) in enumerate(docs, 1):
            source = doc.metadata.get("source_citation", "GigaCorp FAQ")
            content = doc.page_content.replace("**Answer:**", "").strip()
            context_parts.append(
                f"[Document {i}] (Relevance: {score:.3f})\n"
                f"Source: {source}\n"
                f"Content: {content}\n"
            )
        return "\n---\n".join(context_parts)

    def _extract_sources(self, docs: List[tuple]) -> List[Dict[str, Any]]:
        """
        Extract source information from retrieved documents for citations.
        """
        sources = []
        seen = set()

        for doc, score in docs:
            source = doc.metadata.get("source_citation", "GigaCorp FAQ")
            if source not in seen:
                seen.add(source)
                sources.append({
                    "source": source,
                    "relevance_score": round(float(score), 3),
                    "chunk_id": doc.metadata.get("chunk_id", "N/A")
                })

        return sources

    def run(self, question: str, chat_history: str = "") -> Dict[str, Any]:
        """
        Execute the full RAG pipeline: retrieve -> build prompt -> generate -> return with sources.
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.vector_store.similarity_search(question, k=settings.retrieval_k)

        if not retrieved_docs:
            return {
                "answer": "I don't have information about that in our knowledge base. Please contact support@gigacorp.com for assistance.",
                "sources": [],
                "retrieved_count": 0
            }

        # Step 2: Format context
        context = self._format_context(retrieved_docs)

        # Step 3: Build prompt
        prompt = self._build_prompt(question, context, chat_history)

        # Step 4: Generate answer using Groq
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful customer support AI for GigaCorp."},
                {"role": "user", "content": prompt}
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )

        answer = response.choices[0].message.content

        # Step 5: Extract sources
        sources = self._extract_sources(retrieved_docs)

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_count": len(retrieved_docs),
            "model_used": self.model
        }