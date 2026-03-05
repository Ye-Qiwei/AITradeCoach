"""Long-term memory service backed by ChromaDB."""

from __future__ import annotations

import json
from datetime import date
import hashlib
from typing import Any

import chromadb

from ai_trading_coach.config import get_settings
from ai_trading_coach.domain.contracts import (
    MemoryRecallOutput,
    MemoryRecallQuery,
    MemoryWriteInput,
    MemoryWriteOutput,
)
from ai_trading_coach.domain.enums import MemoryStatus, MemoryType
from ai_trading_coach.domain.models import MemoryRecord, RelevantMemorySet


class ChromaLongTermMemoryService:
    """Chroma-backed memory service with metadata-first hybrid retrieval."""

    def __init__(self, persist_dir: str | None = None) -> None:
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        memory_settings = settings.as_memory_settings()
        self.collection_names = {
            MemoryType.RAW_LOG: memory_settings.raw_logs_collection,
            MemoryType.COGNITIVE_CASE: memory_settings.cognitive_cases_collection,
            MemoryType.USER_PROFILE: memory_settings.user_profile_collection,
            MemoryType.ACTIVE_THESIS: memory_settings.active_theses_collection,
            MemoryType.IMPROVEMENT_NOTE: memory_settings.agent_improvement_notes_collection,
        }
        self.collections = {
            memory_type: self.client.get_or_create_collection(name=collection_name)
            for memory_type, collection_name in self.collection_names.items()
        }

    def recall(self, query: MemoryRecallQuery) -> MemoryRecallOutput:
        query_text = " ".join(query.keywords + query.tickers + ([query.regime] if query.regime else [])).strip()
        where = self._build_where_filter(query)

        collected: list[MemoryRecord] = []
        retrieval_notes: list[str] = []
        for memory_type, collection in self.collections.items():
            records = self._query_collection(
                collection=collection,
                memory_type=memory_type,
                query_text=query_text,
                where=where,
                top_k=query.top_k,
                query=query,
                retrieval_notes=retrieval_notes,
            )
            collected.extend(records)

        collected = self._dedupe_records(collected)
        collected.sort(key=lambda record: (record.importance, record.quality_score, record.confidence), reverse=True)
        return MemoryRecallOutput(
            relevant_memories=RelevantMemorySet(records=collected[: query.top_k], retrieval_notes=retrieval_notes)
        )

    def write(self, data: MemoryWriteInput) -> MemoryWriteOutput:
        dedup_count = 0
        merged_count = 0

        grouped: dict[MemoryType, list[MemoryRecord]] = {memory_type: [] for memory_type in MemoryType}
        seen_ids: set[str] = set()
        for record in data.batch.records:
            if record.memory_id in seen_ids:
                dedup_count += 1
                continue
            seen_ids.add(record.memory_id)
            grouped[record.memory_type].append(record)

        written_ids: list[str] = []
        for memory_type, records in grouped.items():
            if not records:
                continue
            collection = self.collections[memory_type]

            ids = [record.memory_id for record in records]
            documents = [record.document_text for record in records]
            metadatas = [self._to_metadata(record) for record in records]
            embeddings = [self._simple_embedding(record.document_text) for record in records]

            collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            written_ids.extend(ids)

        return MemoryWriteOutput(
            written_memory_ids=written_ids,
            dedup_count=dedup_count,
            merged_count=merged_count,
        )

    def _query_collection(
        self,
        collection,
        memory_type: MemoryType,
        query_text: str,
        where: dict[str, Any] | None,
        top_k: int,
        query: MemoryRecallQuery,
        retrieval_notes: list[str],
    ) -> list[MemoryRecord]:
        try:
            if query_text:
                query_embedding = self._simple_embedding(query_text)
                result = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where,
                )
                retrieval_notes.append(f"{collection.name}: embedding+metadata query")
                return self._records_from_query_result(result, memory_type)

            result = collection.get(where=where, limit=top_k)
            retrieval_notes.append(f"{collection.name}: metadata-only retrieval")
            return self._records_from_get_result(result, memory_type)
        except Exception as exc:  # noqa: BLE001
            retrieval_notes.append(f"{collection.name}: retrieval degraded ({exc})")
            fallback = collection.get(where=where, limit=top_k)
            records = self._records_from_get_result(fallback, memory_type)
            return self._keyword_filter(records, query)

    def _records_from_query_result(self, result: dict[str, Any], memory_type: MemoryType) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0] if result.get("distances") else []

        for idx, memory_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            document = docs[idx] if idx < len(docs) else ""
            distance = distances[idx] if idx < len(distances) else None
            records.append(self._record_from_metadata(memory_id, document, metadata, memory_type, distance))
        return records

    def _records_from_get_result(self, result: dict[str, Any], memory_type: MemoryType) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        for idx, memory_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            document = docs[idx] if idx < len(docs) else ""
            records.append(self._record_from_metadata(memory_id, document, metadata, memory_type, None))
        return records

    def _record_from_metadata(
        self,
        memory_id: str,
        document: str,
        metadata: dict[str, Any],
        fallback_memory_type: MemoryType,
        distance: float | None,
    ) -> MemoryRecord:
        memory_type_raw = str(metadata.get("memory_type", fallback_memory_type.value))
        memory_type = MemoryType(memory_type_raw)
        source_date_raw = metadata.get("source_date")
        source_date = date.fromisoformat(source_date_raw) if isinstance(source_date_raw, str) and source_date_raw else None

        quality_score = self._to_float(metadata.get("quality_score"), default=0.5)
        if distance is not None:
            quality_score = max(0.0, min(1.0, 1.0 - float(distance)))

        return MemoryRecord(
            memory_id=memory_id,
            user_id=str(metadata.get("user_id", "unknown")),
            memory_type=memory_type,
            source_date=source_date,
            tickers=self._split_csv(metadata.get("tickers")),
            regime=self._none_if_empty(metadata.get("regime")),
            emotion_tags=self._split_csv(metadata.get("emotion_tags")),
            quality_score=quality_score,
            document_text=document,
            structured_payload=self._json_or_empty(metadata.get("structured_payload")),
            status=MemoryStatus(str(metadata.get("status", MemoryStatus.ACTIVE.value))),
            importance=self._to_float(metadata.get("importance"), default=0.5),
            confidence=self._to_float(metadata.get("confidence"), default=0.5),
            keywords=self._split_csv(metadata.get("keywords")),
            version=int(self._to_float(metadata.get("version"), default=1.0)),
        )

    def _build_where_filter(self, query: MemoryRecallQuery) -> dict[str, Any] | None:
        predicates: list[dict[str, Any]] = [{"user_id": query.user_id}]

        if query.regime:
            predicates.append({"regime": query.regime})

        if query.date_from:
            predicates.append({"source_date": {"$gte": query.date_from.isoformat()}})
        if query.date_to:
            predicates.append({"source_date": {"$lte": query.date_to.isoformat()}})

        if len(predicates) == 1:
            return predicates[0]
        return {"$and": predicates}

    def _to_metadata(self, record: MemoryRecord) -> dict[str, Any]:
        return {
            "user_id": record.user_id,
            "memory_type": record.memory_type.value,
            "source_date": record.source_date.isoformat() if record.source_date else "",
            "tickers": ",".join(record.tickers),
            "regime": record.regime or "",
            "emotion_tags": ",".join(record.emotion_tags),
            "quality_score": float(record.quality_score),
            "structured_payload": json.dumps(record.structured_payload, ensure_ascii=False),
            "status": record.status.value,
            "importance": float(record.importance),
            "confidence": float(record.confidence),
            "keywords": ",".join(record.keywords),
            "version": int(record.version),
        }

    def _keyword_filter(self, records: list[MemoryRecord], query: MemoryRecallQuery) -> list[MemoryRecord]:
        if not query.keywords and not query.tickers and not query.emotion_tags:
            return records

        keywords = {x.lower() for x in query.keywords + query.tickers + query.emotion_tags if x}
        out: list[MemoryRecord] = []
        for record in records:
            haystack = " ".join(
                [
                    record.document_text,
                    " ".join(record.tickers),
                    " ".join(record.keywords),
                    " ".join(record.emotion_tags),
                    record.regime or "",
                ]
            ).lower()
            if any(keyword in haystack for keyword in keywords):
                out.append(record)
        return out

    def _dedupe_records(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        seen: set[str] = set()
        out: list[MemoryRecord] = []
        for record in records:
            if record.memory_id in seen:
                continue
            seen.add(record.memory_id)
            out.append(record)
        return out

    def _split_csv(self, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if not isinstance(value, str):
            return []
        return [part.strip() for part in value.split(",") if part.strip()]

    def _json_or_empty(self, value: Any) -> dict[str, Any]:
        if not value:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}

    def _to_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _none_if_empty(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _simple_embedding(self, text: str, dim: int = 16) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        for idx in range(dim):
            byte = digest[idx % len(digest)]
            values.append(byte / 255.0)
        return values


# Backward-compatible alias
PlaceholderLongTermMemoryService = ChromaLongTermMemoryService
