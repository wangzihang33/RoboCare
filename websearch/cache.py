from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class WebSearchCache:
    CACHE_VERSION = "websearch_v2"

    MARKET_TTL_SECONDS = 12 * 60 * 60
    GENERAL_TTL_SECONDS = 24 * 60 * 60
    EVERGREEN_TTL_SECONDS = 7 * 24 * 60 * 60

    MARKET_KEYWORDS = {
        "2024",
        "2025",
        "2026",
        "latest",
        "price",
        "ranking",
        "rank",
        "compare",
        "comparison",
        "market",
        "\u6700\u65b0",
        "\u4ef7\u683c",
        "\u6392\u884c",
        "\u6392\u540d",
        "\u5bf9\u6bd4",
        "\u5e02\u573a",
        "\u6027\u4ef7\u6bd4",
        "\u54c1\u724c",
    }

    EVERGREEN_KEYWORDS = {
        "how to",
        "why",
        "guide",
        "maintenance",
        "troubleshoot",
        "\u600e\u4e48",
        "\u5982\u4f55",
        "\u4e3a\u4ec0\u4e48",
        "\u6545\u969c",
        "\u7ef4\u62a4",
        "\u4fdd\u517b",
        "\u4f7f\u7528",
    }

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or get_abs_path("cache/websearch"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, query: str, *, top_k: int) -> dict | None:
        meta = self._build_key_meta(query, top_k=top_k)
        cache_path = self._cache_path(meta["cache_key"])
        if not cache_path.exists():
            return None

        try:
            with cache_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"[WebSearchCache] failed to read cache {cache_path}: {exc}")
            return None

        if self._is_expired(payload):
            logger.info(f"[WebSearchCache] cache expired: {meta['cache_key']}")
            return None

        logger.info(f"[WebSearchCache] cache hit: {meta['cache_key']}")
        return payload

    def set(
        self,
        query: str,
        *,
        top_k: int,
        answer: str,
        sources: list[dict],
        trace: dict,
    ) -> dict:
        meta = self._build_key_meta(query, top_k=top_k)
        created_at = self._now()
        expires_at = created_at + timedelta(seconds=meta["ttl_seconds"])
        payload = {
            **meta,
            "query": query,
            "created_at": created_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "answer": answer,
            "sources": sources,
            "trace": trace,
        }

        cache_path = self._cache_path(meta["cache_key"])
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"[WebSearchCache] cache stored: {meta['cache_key']}")
        return payload

    def _build_key_meta(self, query: str, *, top_k: int) -> dict:
        normalized_query = self.normalize_query(query)
        ttl_seconds = self.ttl_seconds_for_query(normalized_query)
        date_bucket = int(time.time() // ttl_seconds)
        raw_key = "|".join(
            [
                self.CACHE_VERSION,
                normalized_query,
                str(top_k),
                str(date_bucket),
            ]
        )
        return {
            "cache_version": self.CACHE_VERSION,
            "cache_key": hashlib.sha256(raw_key.encode("utf-8")).hexdigest(),
            "normalized_query": normalized_query,
            "top_k": top_k,
            "date_bucket": str(date_bucket),
            "ttl_seconds": ttl_seconds,
        }

    @classmethod
    def normalize_query(cls, query: str) -> str:
        return re.sub(r"\s+", " ", query or "").strip().lower()

    @classmethod
    def ttl_seconds_for_query(cls, normalized_query: str) -> int:
        if any(keyword in normalized_query for keyword in cls.MARKET_KEYWORDS):
            return cls.MARKET_TTL_SECONDS
        if any(keyword in normalized_query for keyword in cls.EVERGREEN_KEYWORDS):
            return cls.EVERGREEN_TTL_SECONDS
        return cls.GENERAL_TTL_SECONDS

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    @classmethod
    def _is_expired(cls, payload: dict[str, Any]) -> bool:
        expires_at = payload.get("expires_at")
        if not expires_at:
            return True
        try:
            return datetime.fromisoformat(expires_at) <= cls._now()
        except ValueError:
            return True

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
