from __future__ import annotations

from typing import List, Dict, Any
from loguru import logger
import json

from pydantic import BaseModel, Field


from app.schemas.retrieval import RouterDecision
from app.services.retrieval import agent_state
from app.services.retrieval.vector_store import VectorStoreService
from app.services.utils import get_openai_client
from app.prompts.resolver import RESOLVER_SYSTEM_PROMPT
from app.prompts.rewrite import QUERY_REWRITE_SYSTEM_PROMPT
from app.prompts.router import TOOL_ROUTER_SYSTEM_PROMPT
from app.services.user_db import UserDBService, get_user_db_service


class ToolDecision(BaseModel):
    action: Literal["user_db_only", "vector_db_only", "user_db_then_vector_db", "plain_answer"]
    reason: str = Field(default="")
    user_db_fields: List[str] = Field(default_factory=list)
    search_query_hint: str = Field(default="")


class AgentOrchestrator:
    def __init__(self, vector_store: VectorStoreService, user_db: Optional[UserDBService] = None):
        self.vector_store = vector_store
        self.user_db = user_db or get_user_db_service()

    async def decide_tools(self, *, user_message: str) -> ToolDecision:
        client, model = await get_openai_client()

        available_user_fields = [
            "grade",
            "course_of_study",
            "previous_scores",
            "strengths",
            "weaknesses",
            "recent_topics",
        ]

        system_prompt = TOOL_ROUTER_SYSTEM_PROMPT.format(
            available_user_fields=json.dumps(available_user_fields),
            user_message=user_message,
        )

        res = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw = (res.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(raw)
            return ToolDecision.model_validate(parsed)
        except Exception:
            # Safe fallback: use retrieval
            return ToolDecision(action="vector_db_only", reason="fallback", user_db_fields=[], search_query_hint="")

    async def handle_query(
        self,
        session_id: str,
        user_id: str,
        message: str,
        selected_slugs: List[str],
    ) -> Dict[str, Any]:
        """
        POC:
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

            # Continue with normal tool routing + retrieval path
            message = pending.original_question

        if len(article_ids) != 1:
            return await self.set_uncertain_state(
                session_id=session_id,
                user_id=user_id,
                original_q=message,
                article_ids=article_ids,
                question="Please select exactly one article_id for this question.",
            )

        tool_decision = await self.decide_tools(user_message=message)
        logger.info(f"[tool_router] action={tool_decision.action} reason={tool_decision.reason!r}")

        user_profile: Optional[dict] = None
        if tool_decision.action in ("user_db_only", "user_db_then_vector_db"):
            prof = self.user_db.get_user_profile(user_id=user_id)
            user_profile = prof.data if prof else None

        if tool_decision.action == "plain_answer":
            return {
                "type": "plain_answer",
                "plan": tool_decision.model_dump(),
                "user_profile": user_profile,
                "message": "No retrieval required for this question (POC: answer step not implemented here).",
            }

        if tool_decision.action == "user_db_only":
            return {
                "type": "user_db_result",
                "plan": tool_decision.model_dump(),
                "user_profile": user_profile,
            }

        # vector_db_only OR user_db_then_vector_db
        retrieval = await self.execute_retrieval(
            query=message,
            article_ids=article_ids,
            intent="single",
        )
        retrieval["plan"] = tool_decision.model_dump()
        retrieval["user_profile"] = user_profile
        return retrieval

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