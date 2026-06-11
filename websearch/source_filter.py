from __future__ import annotations

import re
from dataclasses import replace
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from websearch.models import SearchResult, SourceFilterStats


TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "spm",
    "yclid",
}

UNSUPPORTED_EXTENSIONS = {
    ".7z",
    ".avi",
    ".doc",
    ".docx",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".xls",
    ".xlsx",
    ".zip",
}


def normalize_url(url: str) -> str:
    raw_url = (url or "").strip()
    if not raw_url:
        return ""

    parsed = urlparse(raw_url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""

    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not _is_tracking_param(key)
    ]
    query = urlencode(query_pairs, doseq=True)

    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            "",
            query,
            "",
        )
    )


def extract_domain(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def filter_search_results(
    results: list[SearchResult],
    *,
    max_results: int | None = None,
    max_per_domain: int = 2,
) -> tuple[list[SearchResult], SourceFilterStats]:
    stats = SourceFilterStats(original_count=len(results))
    filtered: list[SearchResult] = []
    seen_urls: set[str] = set()
    domain_counts: dict[str, int] = {}

    for result in results:
        normalized_url = normalize_url(result.url)
        if not normalized_url:
            stats.record_drop(url=result.url, reason="invalid_url")
            continue

        if _is_unsupported_url(normalized_url):
            stats.record_drop(url=normalized_url, reason="unsupported_url")
            continue

        if not _has_search_signal(result):
            stats.record_drop(url=normalized_url, reason="low_quality")
            continue

        if normalized_url in seen_urls:
            stats.record_drop(url=normalized_url, reason="duplicate_url")
            continue

        domain = extract_domain(normalized_url)
        if domain_counts.get(domain, 0) >= max_per_domain:
            stats.record_drop(url=normalized_url, reason="duplicate_domain")
            continue

        seen_urls.add(normalized_url)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        filtered.append(
            replace(
                result,
                url=normalized_url,
                original_url=result.original_url or result.url,
                domain=domain,
            )
        )

        if max_results is not None and len(filtered) >= max_results:
            break

    stats.kept_count = len(filtered)
    return filtered, stats


def _is_tracking_param(name: str) -> bool:
    lowered = name.lower()
    return lowered in TRACKING_PARAMS or lowered.startswith("utm_")


def _is_unsupported_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(extension) for extension in UNSUPPORTED_EXTENSIONS)


def _has_search_signal(result: SearchResult) -> bool:
    return bool(result.title.strip() or result.snippet.strip())
