from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    rank: int = 0
    domain: str = ""
    original_url: str = ""
    source: str = "serper"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FetchedPage:
    title: str
    url: str
    snippet: str = ""
    rank: int = 0
    domain: str = ""
    content: str = ""
    fetched_at: str = ""
    status: str = "success"
    error_message: str = ""
    source: str = "serper"

    @classmethod
    def from_search_result(
        cls,
        result: SearchResult,
        *,
        content: str = "",
        status: str = "success",
        error_message: str = "",
        fetched_at: str | None = None,
    ) -> "FetchedPage":
        return cls(
            title=result.title,
            url=result.url,
            snippet=result.snippet,
            rank=result.rank,
            domain=result.domain,
            content=content,
            fetched_at=fetched_at or datetime.now(timezone.utc).isoformat(),
            status=status,
            error_message=error_message,
            source=result.source,
        )

    @property
    def content_length(self) -> int:
        return len(self.content)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["content_length"] = self.content_length
        return payload

    def to_metadata(self) -> dict:
        return {
            "source_id": f"S{self.rank}" if self.rank else "",
            "title": self.title,
            "url": self.url,
            "domain": self.domain,
            "snippet": self.snippet,
            "search_rank": self.rank,
            "fetched_at": self.fetched_at,
            "content_length": self.content_length,
            "source": self.source,
        }


@dataclass
class SourceFilterStats:
    original_count: int = 0
    kept_count: int = 0
    dropped_invalid_url: int = 0
    dropped_unsupported_url: int = 0
    dropped_duplicate_url: int = 0
    dropped_duplicate_domain: int = 0
    dropped_low_quality: int = 0
    drop_reasons: list[dict] = field(default_factory=list)

    def record_drop(self, *, url: str, reason: str) -> None:
        self.drop_reasons.append({"url": url, "reason": reason})
        if reason == "invalid_url":
            self.dropped_invalid_url += 1
        elif reason == "unsupported_url":
            self.dropped_unsupported_url += 1
        elif reason == "duplicate_url":
            self.dropped_duplicate_url += 1
        elif reason == "duplicate_domain":
            self.dropped_duplicate_domain += 1
        elif reason == "low_quality":
            self.dropped_low_quality += 1

    def to_dict(self) -> dict:
        return asdict(self)
