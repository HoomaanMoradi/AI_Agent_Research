"""Transparent proxy metrics; semantic correctness requires independent review."""
import json
import time
from pathlib import Path
from .agent import CITATION


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_evaluation(store, cases: list[dict], agent=None, output: Path = Path("evaluation/results.json"), retrieval_k: int = 5) -> dict:
    results = []
    for case in cases:
        start = time.monotonic()
        hits = store.search(case.get("retrieval_query", case["question"]), retrieval_k)
        expected = {(e["paper_id"], int(e["page"])) for e in case["evidence"]}
        retrieved = {(h["paper_id"], int(h["page"])) for h in hits}
        row = {"id": case["id"], "question": case["question"], "expected_evidence": case["evidence"],
               "retrieved": [{"paper_id": h["paper_id"], "page": h["page"], "text": h["text"]} for h in hits],
               f"page_recall_at_{retrieval_k}": len(expected & retrieved)/len(expected) if expected else None,
               f"page_precision_at_{retrieval_k}": len(expected & retrieved)/len(retrieved) if expected and retrieved else (0 if expected else None)}
        if agent:
            try:
                answer = agent.ask(case["question"])
                cites = {(pid, int(page)) for pid, page in CITATION.findall(answer["answer"])}
                seen = {tuple(s) for s in answer["sources"]}
                selected = {t["tool"] for t in answer["trace"] if t["ok"]}
                completed = answer["stop_reason"] == "completed"
                row.update({"agent": answer,
                    "citation_provenance_precision": len(cites & seen)/len(cites) if cites else None,
                    "expected_citation_recall": len(cites & expected)/len(expected) if expected else None,
                    "required_tool_coverage": len(selected & set(case["required_tools"]))/len(case["required_tools"]) if case["required_tools"] else None,
                    "abstention_match": ("insufficient evidence" in answer["answer"].lower()) == case["unanswerable"] if completed else None,
                    "answer_key_match_proxy": all(term.casefold() in answer["answer"].casefold() for term in case.get("answer_terms", [])) if completed and not case["unanswerable"] else None,
                    "manual_correctness": None, "manual_evidence_support": None})
            except Exception as exc:
                row["agent_error"] = type(exc).__name__
        row["seconds"] = round(time.monotonic()-start, 6)
        results.append(row)
    report = {"mode": "live_agent" if agent else "retrieval_only", "retrieval_k": retrieval_k, "created_unix": time.time(),
              "corpus": [p.model_dump() for p in store.papers()], "results": results,
              "manual_review": "Pending. Automatic metrics are lexical/page/tool proxies, not proof of scientific correctness."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report
