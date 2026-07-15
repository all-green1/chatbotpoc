from typing import List, Dict, Any
from loguru import logger
import json

from app.schemas.retrieval import RouterDecision
from app.services.retrieval import agent_state
from app.services.retrieval import VectorStoreService
from app.services.retrieval import get_book_brief, get_openai_client
from app.prompts import ROUTER_SYSTEM_PROMPT, RESOLVER_SYSTEM_PROMPT, QUERY_REWRITE_SYSTEM_PROMPT
from app.schemas.books import SearchChunkResult

class AgentOrchestrator:
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store

    async def handle_query(
        self,
        session_id: str,
        user_id: str,
        message: str,
        selected_slugs: List[str]
    ) -> Dict[str, Any]:
        """
        Main gateway for processing queries with 1 or 2 selected books.
        """
        # 1. Handle Multi-turn State Machine
        pending = agent_state.get_pending(session_id, user_id=user_id)

        if pending:
            logger.info(f"Resolving clarification for session {session_id}")
            decision = await self.resolve_clarification(
                original_q=pending.original_question,
                clarifying_question=pending.clarifying_question,
                clarification=message,
                slugs=selected_slugs
            )
            agent_state.clear_pending(session_id)

            if decision.intent == "uncertain":
                # Fallback if still unclear
                return await self.set_uncertain_state(
                    session_id,
                    user_id,
                    pending.original_question,
                    selected_slugs,
                    decision.clarifying_question or pending.clarifying_question,
                )

            return await self.execute_retrieval(pending.original_question, decision.targets, decision.intent)

        # 2. Direct path for single book
        if len(selected_slugs) == 1:
            return await self.execute_retrieval(message, selected_slugs, "single")

        # 3. Router Agent for 2 books
        decision = await self.route_intent(message, selected_slugs)

        if decision.intent == "uncertain":
            return await self.set_uncertain_state(
                session_id, user_id, message, selected_slugs, decision.clarifying_question
            )

        return await self.execute_retrieval(message, decision.targets, decision.intent)

    async def route_intent(self, message: str, slugs: List[str]) -> RouterDecision:
        client, model = await get_openai_client()
        briefs = [await get_book_brief(self.vector_store, s) for s in slugs]
        books_info = "\n\n".join(
            [
                "BOOK\n"
                f"- slug: {b['slug']}\n"
                f"- title: {b['title']}\n"
                f"- language: {b['language'] or 'unknown'}\n"
                f"- summary:\n{b['summary']}\n"
                f"- table_of_contents:\n{b['toc']}\n"
                for b in briefs
            ]
        )
        system_prompt = ROUTER_SYSTEM_PROMPT.format(
            books_info=books_info,
            selected_slugs=slugs,
        )
        res = await client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            response_format=RouterDecision,
        )
        return res.choices[0].message.parsed

    async def resolve_clarification(
        self,
        *,
        original_q: str,
        clarifying_question: str,
        clarification: str,
        slugs: list[str],
    ) -> RouterDecision:
        """
        Resolves a user's clarification or choice in response to a question.
        """
        client, model = await get_openai_client()
        system_prompt = RESOLVER_SYSTEM_PROMPT.format(
            clarifying_question=clarifying_question,
            original_question=original_q,
            user_answer=clarification,
            selected_slugs=slugs,
        )
        res = await client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "system", "content": system_prompt}],
            response_format=RouterDecision
        )
        return res.choices[0].message.parsed

    async def rewrite_for_retrieval(self, *, user_message: str) -> str:
        """
        Uses the normal model to rewrite the user's message into a vector-search-friendly query.
        The original user message remains authoritative for routing/clarification and final answering.
        """
        client, model = await get_openai_client()

        payload = {
            "user_message": user_message,
        }

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

    async def execute_retrieval(self, query: str, slugs: List[str], intent: str):
        search_query = await self.rewrite_for_retrieval(
            user_message=query,
        )

        logger.info(
            f"original_query={query[:120]!r} rewritten_query={search_query[:120]!r}"
        )
        results = await self.vector_store.search_documents(
            collection_name="new books",
            query=search_query,
            slug=slugs
        )

        # Evidence gating ONLY for multi-book
        gated_points = results
        notes: List[str] = []

        if len(slugs) > 1:
            evidence_map = {s: [] for s in slugs}
            for p in results:
                evidence_map[p.payload["book_slug"]].append(p)

            gated_points = []
            for slug in slugs:
                book_chunks = evidence_map[slug]
                # Threshold: Top result must be >= 0.7 to be considered relevant
                if not book_chunks or max(p.score for p in book_chunks) < 0.7:
                    notes.append(
                        f"Book '{slug}' does not contain sufficient direct evidence for this question."
                    )
                else:
                    gated_points.extend(book_chunks)

        if gated_points:
            top_point = max(gated_points, key=lambda c: float(getattr(c, "score", 0.0) or 0.0))
            top_score = float(getattr(top_point, "score", 0.0) or 0.0)

            payload = getattr(top_point, "payload", None) or {}
            content = str(payload.get("content") or "")
            logger.info(
                "[retrieve] top_chunk"
                f" score={top_score:.4f}"
                f" content_preview={content[:240]!r}"
            )
        else:
            logger.info("[retrieve] top_chunk (none) gated_points=0")

        # Normalize to a stable, rich schema (similar to /relevant)
        gated_chunks: List[Dict[str, Any]] = []
        for p in gated_points:
            payload = p.payload or {}

            chunk = SearchChunkResult(
                breadcrumbs=" > ".join(payload.get("breadcrumbs", []) or []),
                page_start=int(payload.get("page_start") or 0),
                page_end=int(payload.get("page_end") or 0),
                content=str(payload.get("content") or ""),
                confidence=p.score,
            ).model_dump()

            # Preserve extra details that are commonly needed downstream
            chunk["book_slug"] = payload.get("book_slug")
            chunk["chunk_id"] = payload.get("id")
            chunk["section_ancestor_ids"] = payload.get("section_ancestor_ids")
            gated_chunks.append(chunk)

        return {
            "type": "retrieval_result",
            "chunks": gated_chunks,
            "notes": notes,
            "intent": intent,
            "query": query
        }

    async def set_uncertain_state(self, session_id, user_id, original_q, slugs, question=None):
        clarifying_q = question
        agent_state.set_pending(
            session_id=session_id,
            user_id=user_id,
            kind="book_or_mode",
            question=original_q,
            slugs=slugs,
            clarification=clarifying_q
        )
        return {"type": "clarification", "message": clarifying_q}
