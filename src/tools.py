import base64
import json
import logging
import time
from collections.abc import Callable
from .calculator import calculate
from .config import Config
from .models import Search, Acquire, Retrieve, Figure, Tables, Calculate, ToolResult
from .processing import Processor
from .literature import Arxiv
from .storage import Store

log = logging.getLogger(__name__)
SPECS = {
    "literature_search": (Search, "Discover arXiv papers. Saves metadata only; acquire selected papers to read evidence."),
    "paper_acquisition": (Acquire, "Download and index a previously discovered accessible arXiv paper."),
    "paper_retrieval": (Retrieve, "Search local paper text. Returns page citations; try alternate keywords if empty."),
    "figure_inspection": (Figure, "Inspect a source page using vision, including vector figures and captions. Requires an ingested PDF."),
    "table_access": (Tables, "Return detected table cells and caption candidates for one paper page."),
    "python_analysis": (Calculate, "Evaluate restricted Python numeric expressions over supplied data. Supports sum, mean, median, min, max, len, sqrt, abs, round. No statements/imports/files. Cite original input evidence separately."),
}


class Toolset:
    def __init__(self, store: Store, arxiv: Arxiv, processor: Processor, vision: Callable | None = None, local_only: bool | None = None, config: Config | None = None):
        self.config = config or Config()
        local_only = self.config.local_only if local_only is None else local_only
        self.store, self.arxiv, self.processor, self.vision = store, arxiv, processor, vision
        self.allowed = set(SPECS) - ({"literature_search", "paper_acquisition"} if local_only else set())
        self.sources: set[tuple[str, int]] = set()
        self.trace: list[dict] = []

    def schemas(self) -> list[dict]:
        schemas = []
        for name, (model, description) in SPECS.items():
            if name not in self.allowed:
                continue
            parameters = model.model_json_schema()
            properties = parameters["properties"]
            if name in {"literature_search", "paper_retrieval"}:
                settings = self.config.discovery if name == "literature_search" else self.config.retrieval
                default = settings.tool_limit if name == "literature_search" else settings.limit
                properties["limit"] = {"type": "integer", "minimum": 1, "maximum": settings.max_tool_limit, "default": default}
                properties["query"]["maxLength"] = self.config.retrieval.max_query_chars
            elif name == "figure_inspection":
                properties["question"]["maxLength"] = self.config.api.max_vision_question_chars
            elif name == "python_analysis":
                properties["expression"]["maxLength"] = self.config.calculator.max_expression_chars
            schemas.append({"type": "function", "name": name, "description": description,
                            "parameters": parameters, "strict": False})
        return schemas

    def execute(self, name: str, arguments: str) -> dict:
        start = time.monotonic()
        source_ids = []
        args = None
        try:
            if name not in self.allowed:
                raise ValueError("Unknown or disabled tool")
            args = SPECS[name][0].model_validate_json(arguments)
            if isinstance(args, (Search, Retrieve)):
                if len(args.query) > self.config.retrieval.max_query_chars:
                    raise ValueError("Query exceeds configured character limit")
                settings = self.config.discovery if isinstance(args, Search) else self.config.retrieval
                default = settings.tool_limit if isinstance(args, Search) else settings.limit
                args = args.model_copy(update={"limit": args.limit if args.limit is not None else default})
                if args.limit > settings.max_tool_limit:
                    raise ValueError("Requested results exceed configured tool limit")
            if isinstance(args, Search):
                result = {"papers": [p.model_dump() for p in self.arxiv.search(args.query, args.limit)], "warnings": list(self.arxiv.warnings)}
                source_ids = [p["id"] for p in result["papers"]]
            elif isinstance(args, Acquire):
                paper = self.store.paper(args.paper_id)
                self.arxiv.download(paper)
                result = self.processor.ingest(paper.id)
                source_ids = [paper.id]
            elif isinstance(args, Retrieve):
                hits = self.store.search(args.query, args.limit)
                self.sources.update((h["paper_id"], int(h["page"])) for h in hits)
                source_ids = [h["citation"] for h in hits]
                result = {"passages": hits, "untrusted_document_data": True}
            elif isinstance(args, (Figure, Tables)):
                paper = self.store.paper(args.paper_id)
                page = self.store.page(paper.id, args.page)
                citation = f"[{paper.id} p.{args.page}]"
                if isinstance(args, Tables):
                    result = {"tables": json.loads(page["tables_json"]), "extraction_status": "unverified candidates; empty or malformed tables require figure_inspection", "source_page_text": page["text"][:self.config.processing.table_text_chars], "figures_captions": json.loads(page["figures_json"]), "citation": citation, "untrusted_document_data": True}
                else:
                    if len(args.question) > self.config.api.max_vision_question_chars:
                        raise ValueError("Vision question exceeds configured character limit")
                    if self.vision is None:
                        raise ValueError("Vision client unavailable; configure OPENROUTER_API_KEY")
                    image = base64.b64encode(self.processor.render(paper.id, args.page)).decode()
                    result = {"inspection": self.vision(image, args.question), "citation": citation, "untrusted_document_data": True}
                self.sources.add((paper.id, args.page))
                source_ids = [citation]
            else:
                result = calculate(args.expression, args.data, settings=self.config.calculator.model_dump())
            output = {"ok": "error" not in result, "result": result}
        except Exception as exc:
            # Do not log provider exception text: it may contain request headers or credentials.
            safe_message = str(exc)[:240] if isinstance(exc, (ValueError, KeyError)) else "Check network, PDF availability, credentials, and configured limits"
            output = {"ok": False, "error": {"type": type(exc).__name__, "message": safe_message}}
        event = {"tool": name, "arguments": "<invalid or redacted>" if args is None else args.model_dump(),
                 "ok": output["ok"], "sources": source_ids, "seconds": round(time.monotonic()-start, 3)}
        self.trace.append(event)
        log.info(json.dumps(event))
        return ToolResult.model_validate(output).model_dump(exclude_none=True)
