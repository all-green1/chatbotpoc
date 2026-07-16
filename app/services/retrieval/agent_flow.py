from __future__ import annotations

from typing import List, Dict, Any
from loguru import logger
import json

from app.schemas.retrieval import RouterDecision
from app.services.retrieval import agent_state
from app.services.retrieval.vector_store import VectorStoreService
from app.services.retrieval.utils import get_openai_client
from app.prompts import RESOLVER_SYSTEM_PROMPT, QUERY_REWRITE_SYSTEM_PROMPT


class AgentOrchestrator:
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store

    async def handle_query(
        self,
        session_id: str,
        user_id: str,
        message: str,
        selected_slugs: List[str],
    ) -> Dict[str, Any]:
        """
        POC: `selected_slugs` is treated as selected_article_ids for backwards compatibility.
        Exactly ONE id is supported.
        """
        article_ids = selected_slugs

        pending = agent_state.get_pending(session_id, user_id=user_id)

        if pending:
            logger.info(f"Resolving clarification for session {session_id}")
            decision = await self.resolve_clarification(
                original_q=pending.original_question,
                clarifying_question=pending.clarifying_question,
                clarification=message,
                article_ids=article_ids,
            )
            agent_state.clear_pending(session_id)

            if decision.intent == "uncertain":
                return await self.set_uncertain_state(
                    session_id=session_id,
                    user_id=user_id,
                    original_q=pending.original_question,
                    article_ids=article_ids,
                    question=decision.clarifying_question or pending.clarifying_question,
                )

            return await self.execute_retrieval(
                query=pending.original_question,
                article_ids=article_ids,
                intent=decision.intent,
            )

        if len(article_ids) != 1:
            return await self.set_uncertain_state(
                session_id=session_id,
                user_id=user_id,
                original_q=message,
                article_ids=article_ids,
                question="Please select exactly one article_id for this question.",
            )

        return await self.execute_retrieval(query=message, article_ids=article_ids, intent="single")

    async def resolve_clarification(
        self,
        *,
        original_q: str,
        clarifying_question: str,
        clarification: str,
        article_ids: list[str],
    ) -> RouterDecision:
        client, model = await get_openai_client()
        system_prompt = RESOLVER_SYSTEM_PROMPT.format(
            clarifying_question=clarifying_question,
            original_question=original_q,
            user_answer=clarification,
            selected_slugs=article_ids,
        )
        res = await client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "system", "content": system_prompt}],
            response_format=RouterDecision,
        )
        return res.choices[0].message.parsed

    async def rewrite_for_retrieval(self, *, user_message: str) -> str:
        client, model = await get_openai_client()

        payload = {"user_message": user_message}

        res = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw = (res.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(raw)
            sq = str(parsed.get("search_query", "")).strip()
            return sq
        except Exception:
            return ""

    async def execute_retrieval(
        self,
        *,
        query: str,
        article_ids: List[str],
        intent: str,
    ) -> Dict[str, Any]:
        if len(article_ids) != 1:
            return {"type": "clarification", "message": "Please select exactly one article_id for retrieval."}

        article_id = (article_ids[0] or "").strip()
        if not article_id:
            return {"type": "clarification", "message": "Selected article_id is empty."}

        search_query = await self.rewrite_for_retrieval(user_message=query)

        logger.info(
            f"original_query={query[:120]!r} rewritten_query={search_query[:120]!r} article_id={article_id!r}"
        )

        results = await self.vector_store.search_documents(
            collection_name="article",
            query=search_query,
            article_id=article_id,
        )

        chunks: List[Dict[str, Any]] = []
        for p in results:
            payload = getattr(p, "payload", None) or {}
            chunks.append(
                {
                    "chunk_id": getattr(p, "id", None) or payload.get("id"),
                    "content": str(payload.get("content") or ""),
                    "confidence": float(getattr(p, "score", 0.0) or 0.0),
                    "article_id": payload.get("article_id"),
                    "source_type": payload.get("source_type"),
                    "tags": payload.get("tags") or [],
                }
            )

        return {
            "type": "retrieval_result",
            "chunks": chunks,
            "notes": [],
            "intent": intent,
            "query": query,
            "article_id": article_id,
        }

    async def set_uncertain_state(
        self,
        *,
        session_id: str,
        user_id: str,
        original_q: str,
        article_ids: List[str],
        question: str | None = None,
    ) -> Dict[str, Any]:
        clarifying_q = question or "Please select exactly one article_id."
        agent_state.set_pending(
            session_id=session_id,
            user_id=user_id,
            kind="article_id_required",
            question=original_q,
            article_ids=article_ids,
            clarification=clarifying_q,
        )
        return {"type": "clarification", "message": clarifying_q}