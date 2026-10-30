"""arXiv only; sequential, cached, rate-limited requests with bounded retries."""
import hashlib
import json
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import httpx
from .models import Paper, stable_id
from .storage import Store
from .config import DiscoverySettings


class Arxiv:
    def __init__(self, store: Store, client: httpx.Client | None = None, delay: float | None = None, settings: DiscoverySettings | None = None):
        self.settings = settings or DiscoverySettings()
        self.store = store
        self.client = client or httpx.Client(timeout=self.settings.timeout_seconds, follow_redirects=False,
            headers={"User-Agent": self.settings.user_agent})
        self.delay = self.settings.request_interval_seconds if delay is None else delay
        self.last_request = 0.0
        self.deadline = float("inf")
        self.warnings: list[dict] = []

    def request(self, url: str, params: dict | None = None, max_bytes: int | None = None) -> bytes:
        max_bytes = self.settings.max_pdf_bytes if max_bytes is None else max_bytes
        for attempt in range(self.settings.attempts):
            pause = max(0, self.delay - (time.monotonic() - self.last_request))
            if time.monotonic() + pause >= self.deadline:
                raise TimeoutError("arXiv execution deadline reached")
            time.sleep(pause)
            self.last_request = time.monotonic()
            try:
                with self.client.stream("GET", url, params=params, timeout=min(self.settings.timeout_seconds, self.deadline-time.monotonic())) as response:
                    if response.status_code == 429 or response.status_code >= 500:
                        retry = response.headers.get("Retry-After", "")
                        try:
                            pause = float(retry)
                        except ValueError:
                            try:
                                pause = (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds()
                            except (ValueError, TypeError):
                                pause = self.settings.retry_backoff_base ** attempt
                        if attempt == self.settings.attempts - 1:
                            response.raise_for_status()
                        if pause > self.settings.max_retry_after_seconds:
                            response.raise_for_status()  # Do not retry earlier than a long server cooldown.
                        time.sleep(min(max(self.delay, pause), max(0, self.deadline-time.monotonic())))
                        continue
                    response.raise_for_status()
                    body = bytearray()
                    for block in response.iter_bytes():
                        if time.monotonic() >= self.deadline:
                            raise TimeoutError("arXiv download deadline reached")
                        body.extend(block)
                        if len(body) > max_bytes:
                            raise ValueError("arXiv response exceeds size limit")
                    return bytes(body)
            except httpx.TransportError:
                if attempt == self.settings.attempts - 1:
                    raise
                time.sleep(min(self.settings.retry_backoff_base ** attempt, max(0, self.deadline-time.monotonic())))
        raise RuntimeError("arXiv retries exhausted")

    def search(self, query: str, limit: int = 5) -> list[Paper]:
        self.warnings.clear()
        cache = self.store.root / "cache"
        cache.mkdir(exist_ok=True)
        key = hashlib.sha256(json.dumps([query, limit, self.settings.sort_by, self.settings.sort_order, self.settings.page_size, self.settings.scan_multiplier]).encode()).hexdigest()
        path = cache / (key + ".json")
        if path.exists() and time.time() - path.stat().st_mtime < self.settings.cache_ttl_seconds:
            cached = [Paper.model_validate(p) for p in json.loads(path.read_text())]
            return [self.store.paper(self.store.save(p)) for p in cached]
        papers: dict[str, Paper] = {}
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for start in range(0, limit * self.settings.scan_multiplier, min(self.settings.page_size, limit)):
            try:
                body = self.request("https://export.arxiv.org/api/query", {"search_query": query, "start": start, "max_results": min(self.settings.page_size, limit), "sortBy": self.settings.sort_by, "sortOrder": self.settings.sort_order}, self.settings.max_metadata_bytes)
                root = ET.fromstring(body)
            except (httpx.HTTPError, ET.ParseError, TimeoutError) as exc:
                if not papers:
                    raise
                self.warnings.append({"start": start, "error": type(exc).__name__, "partial": True})
                break
            entries = root.findall("a:entry", ns)
            for entry in entries:
                get = lambda name: " ".join((entry.findtext("a:" + name, "", ns)).split())
                pid = stable_id(get("id"))  # Also rejects arXiv error-feed identifiers.
                pdf = next((link.attrib["href"] for link in entry.findall("a:link", ns) if link.attrib.get("title") == "pdf"), None)
                p = Paper(id=pid, title=get("title"), authors=[a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)], published=get("published"), abstract=get("summary"), url=f"https://arxiv.org/abs/{pid}", pdf_url=pdf)
                canonical = self.store.save(p)
                papers[canonical] = self.store.paper(canonical)
                if len(papers) >= limit:
                    break
            if len(papers) >= limit or len(entries) < min(self.settings.page_size, limit):
                break
        result = list(papers.values())
        if not self.warnings:
            path.write_text(json.dumps([p.model_dump() for p in result]))
        return result

    def download(self, paper: Paper) -> Path:
        path = self.store.pdf_path(paper.id)
        if path.exists() and path.read_bytes().startswith(b"%PDF-"):
            return path
        if not paper.pdf_url:
            raise ValueError("No publicly accessible PDF listed")
        parsed = urlparse(paper.pdf_url)
        if parsed.hostname not in {"arxiv.org", "export.arxiv.org"} or not parsed.path.startswith("/pdf/"):
            raise ValueError("PDF URL must be an arXiv PDF")
        # Canonical HTTPS endpoint; no arbitrary URL fetching or redirects.
        versioned = parsed.path.removeprefix("/pdf/").removesuffix(".pdf")
        if stable_id(versioned) != paper.id:
            raise ValueError("PDF identifier does not match paper metadata")
        body = self.request(f"https://arxiv.org/pdf/{versioned}")
        if not body.startswith(b"%PDF-"):
            raise ValueError("arXiv returned a non-PDF response")
        path.parent.mkdir(exist_ok=True)
        temp = path.with_suffix(".part")
        temp.write_bytes(body)
        temp.replace(path)
        return path
