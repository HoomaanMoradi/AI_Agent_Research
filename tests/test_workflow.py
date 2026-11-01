"""Offline integration coverage; scientific text below is a synthetic fixture."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pymupdf
import pytest

from research_agent.agent import Agent
from research_agent.calculator import calculate
from research_agent.config import Config
from research_agent.literature import Arxiv
from research_agent.models import Paper
from research_agent.processing import Processor
from research_agent.storage import Store
from research_agent.tools import Toolset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def corpus(tmp_path):
    store = Store(tmp_path / "data")
    paper = Paper(id="2401.00001", title="Synthetic fixture", authors=["Test Author"],
                  published="2024-01-01", abstract="Test only", url="https://arxiv.org/abs/2401.00001",
                  pdf_url="https://arxiv.org/pdf/2401.00001")
    store.save(paper)
    path = store.pdf_path(paper.id)
    path.parent.mkdir()
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "1 Results\nSynthetic model accuracy was 82 percent.\nThis is a test fixture.")
        doc.save(path)
    yield store, Processor(store), paper
    store.db.close()


def cli(*args):
    return subprocess.run([sys.executable, "-m", "research_agent", *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=20)


def test_installed_entry_points():
    command = Path(sys.executable).parent / "research-agent"
    result = subprocess.run([str(command), "--help"], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "collect" in result.stdout
    result = cli("config")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["query"] == 'all:"small language models"'


def test_pdf_retrieval_render_and_reindex(corpus):
    store, processor, paper = corpus
    assert processor.ingest(paper.id)["pages"] == 1
    assert processor.ingest(paper.id)["cached"] is True
    assert processor.ingest(paper.id, force=True)["cached"] is False
    hits = store.search("accuracy")
    assert len(hits) == 1
    assert hits[0]["citation"] == "[2401.00001 p.1]"
    assert "82 percent" in hits[0]["text"]
    assert processor.render(paper.id, 1).startswith(b"\x89PNG")


def test_cli_index_retrieve_evaluate(corpus, tmp_path):
    store, _, _ = corpus
    for args in [("papers",), ("index",), ("retrieve", "accuracy")]:
        result = cli("--data", store.root, *args)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)
    cases = tmp_path / "questions.jsonl"
    cases.write_text(json.dumps({"id": "fixture", "question": "accuracy",
                                 "evidence": [{"paper_id": "2401.00001", "page": 1}]}) + "\n")
    result = cli("--data", store.root, "evaluate", "--cases", cases, "--output", tmp_path / "report.json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["results"][0]["page_recall_at_5"] == 1


def test_calculator_worker():
    assert calculate("mean(scores)", {"scores": [2, 4, 6]})["value"] == 4
    assert "error" in calculate("__import__('os')", {})
    assert "error" in calculate("1 / 0", {})


def test_tools_and_agent(corpus):
    store, processor, paper = corpus
    processor.ingest(paper.id)
    config = Config(local_only=True, max_tokens=60000)
    arxiv = Arxiv(store)
    tools = Toolset(store, arxiv, processor, config=config)
    assert "literature_search" not in {s["name"] for s in tools.schemas()}
    assert not tools.execute("literature_search", '{"query":"accuracy"}')["ok"]
    assert tools.execute("table_access", '{"paper_id":"2401.00001","page":1}')["ok"]

    class Model:
        calls = 0

        def respond(self, history, schemas, output_tokens, timeout):
            self.calls += 1
            usage = SimpleNamespace(input_tokens=50, output_tokens=20)
            if self.calls == 1:
                call = SimpleNamespace(type="function_call", name="paper_retrieval",
                                       arguments='{"query":"accuracy"}', call_id="test-call")
                return SimpleNamespace(usage=usage, output=[call])
            assert json.loads(history[-1]["output"])["ok"]
            return SimpleNamespace(usage=usage, output=[], status="completed",
                                   output_text="Reported findings: synthetic accuracy was 82 percent [2401.00001 p.1].")

    answer = Agent(config, Model(), tools).ask("What accuracy was reported?")
    assert answer["stop_reason"] == "completed"
    assert answer["tool_calls"] == 1
    assert answer["sources"] == [(paper.id, 1)]
    arxiv.client.close()


def test_arxiv_search_download_and_cache(tmp_path):
    store = Store(tmp_path / "corpus")
    requests = []
    feed = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
    <id>http://arxiv.org/abs/2401.00001v1</id><title>Synthetic fixture</title>
    <author><name>Test Author</name></author><published>2024-01-01</published>
    <summary>Test only</summary><link title="pdf" href="http://arxiv.org/pdf/2401.00001v1"/>
    </entry></feed>'''

    def handle(request):
        requests.append(request)
        return httpx.Response(200, content=feed if request.url.path == "/api/query" else b"%PDF-fixture")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        arxiv = Arxiv(store, client=client, delay=0)
        papers = arxiv.search("fixture", 1)
        assert arxiv.search("fixture", 1) == papers
        assert len(requests) == 1
        assert arxiv.download(papers[0]).read_bytes() == b"%PDF-fixture"
        assert arxiv.download(papers[0]).exists()
        assert len(requests) == 2
    store.db.close()


def test_invalid_config_and_limit(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("topic: one\ntopic: two\n")
    with pytest.raises(ValueError, match="Invalid YAML"):
        Config.from_yaml(path)
    result = cli("--data", tmp_path / "data", "retrieve", "accuracy", "--limit", "0")
    assert result.returncode == 1
    assert "--limit" in result.stderr
