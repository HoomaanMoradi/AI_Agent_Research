"""Command-line entry point: YAML defaults, then explicit command overrides."""
import argparse
import hashlib
import json
import logging
from pathlib import Path

from .agent import Agent, OpenRouterModel
from .config import Config
from .evaluation import load_cases, run_evaluation
from .literature import Arxiv
from .logging import JsonFormatter
from .models import Paper
from .processing import Processor
from .storage import Store
from .tools import Toolset


def main() -> None:
    parser = argparse.ArgumentParser(description="arXiv scientific research agent")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="YAML settings file")
    parser.add_argument("--data", type=Path, help="Override the configured corpus directory")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("config", help="Validate YAML and show effective non-secret settings")
    collect = subs.add_parser("collect", help="Discover and optionally ingest arXiv papers")
    collect.add_argument("--query")
    collect.add_argument("--limit", type=int)
    collect.add_argument("--metadata-only", action=argparse.BooleanOptionalAction, default=None)
    collect.add_argument("--manifest", type=Path, help="Replay a pinned arXiv corpus manifest")
    index = subs.add_parser("index", help="Index downloaded papers")
    index.add_argument("--force", action=argparse.BooleanOptionalAction, default=None)
    subs.add_parser("papers", help="List saved paper metadata")
    retrieve = subs.add_parser("retrieve", help="Search passages without a model request")
    retrieve.add_argument("query")
    retrieve.add_argument("--limit", type=int)
    ask = subs.add_parser("ask", help="Run model-directed research tools")
    ask.add_argument("question")
    ask.add_argument("--output", type=Path)
    ask.add_argument("--local-only", action=argparse.BooleanOptionalAction, default=None)
    evaluate = subs.add_parser("evaluate")
    evaluate.add_argument("--cases", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--live", action=argparse.BooleanOptionalAction, default=None)
    evaluate.add_argument("--local-only", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    try:
        config = Config.from_yaml(args.config)
        if args.data is not None:
            config = config.model_copy(update={"data": args.data.resolve()})
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logging.basicConfig(level=config.logging.level, handlers=[handler])
        logging.getLogger("httpx").setLevel(config.logging.http_level)
        if args.command == "config":
            print(config.model_dump_json(indent=2))
            return
        store = Store(config.data, config.retrieval)
        arxiv = Arxiv(store, settings=config.discovery)
        processor = Processor(store, config.processing)
        result = None
        if args.command == "collect":
            limit = args.limit if args.limit is not None else config.paper_limit
            if not 1 <= limit <= 100:
                raise ValueError("--limit must be between 1 and 100")
            if args.manifest:
                manifest = json.loads(args.manifest.read_text())
                papers = []
                for item in manifest["papers"]:
                    paper = Paper.model_validate(item["metadata"])
                    papers.append(store.paper(store.save(paper)))
            else:
                papers = arxiv.search(args.query if args.query is not None else config.query, limit)
            result = {"papers": [p.model_dump() for p in papers], "ingestion": [], "warnings": list(arxiv.warnings)}
            metadata_only = config.metadata_only if args.metadata_only is None else args.metadata_only
            if not metadata_only:
                for paper in papers:
                    try:
                        arxiv.download(paper)
                        if args.manifest:
                            expected = next(item["sha256"] for item in manifest["papers"] if item["metadata"]["id"] == paper.id)
                            if hashlib.sha256(store.pdf_path(paper.id).read_bytes()).hexdigest() != expected:
                                raise ValueError("PDF checksum differs from pinned corpus manifest")
                        result["ingestion"].append(processor.ingest(paper.id))
                    except Exception as exc:
                        result["ingestion"].append({"paper_id": paper.id, "error": type(exc).__name__,
                            "detail": str(exc)[:200] if isinstance(exc, ValueError) else "PDF unavailable or processing failed",
                            "action": "Retry collect; use a fresh --data directory if a manifest checksum differs"})
        elif args.command == "index":
            force = config.force_reindex if args.force is None else args.force
            result = []
            for paper in store.papers():
                try:
                    result.append(processor.ingest(paper.id, force=force))
                except Exception as exc:
                    result.append({"paper_id": paper.id, "error": type(exc).__name__})
        elif args.command == "papers":
            result = [p.model_dump() for p in store.papers()]
        elif args.command == "retrieve":
            limit = config.retrieval.limit if args.limit is None else args.limit
            if not 1 <= limit <= config.retrieval.max_tool_limit:
                raise ValueError("--limit must be within retrieval.max_tool_limit")
            result = store.search(args.query, limit)
        else:
            is_ask = args.command == "ask"
            live = is_ask or (config.evaluation.live if args.live is None else args.live)
            local_default = config.local_only if is_ask else config.evaluation.local_only
            local_only = local_default if args.local_only is None else args.local_only
            agent = None
            if live:
                model = OpenRouterModel(config)
                toolset = Toolset(store, arxiv, processor, model.inspect, local_only=local_only, config=config)
                agent = Agent(config, model, toolset)
            if is_ask:
                result = agent.ask(f"Research topic: {config.topic}\n{args.question}")
                output = args.output or config.answer_output
                if output:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(result, indent=2))
            else:
                result = run_evaluation(store, load_cases(args.cases or config.evaluation.cases), agent,
                                        args.output or config.evaluation.output, config.evaluation.retrieval_k)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        message = str(exc) if isinstance(exc, ValueError) else "Check connectivity, installed dependencies, YAML paths, corpus files, and provider configuration."
        parser.exit(1, f"{type(exc).__name__}: {message}\n")
