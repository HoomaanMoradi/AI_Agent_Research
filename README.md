# AI Agent Research

A command-line scientific research assistant that discovers arXiv papers, downloads and indexes their PDFs, and answers questions using page-level evidence. It stores a local corpus in SQLite with FTS5 search and uses an OpenAI-compatible Responses API through OpenRouter for model-directed research.

The agent can search literature, acquire papers, retrieve passages, inspect source pages with a vision model, access extracted tables, and calculate numeric expressions. Answers include citations such as `[1706.03762 p.3]`; citation validation checks that cited pages were returned by tools during that question.

## Requirements

- Python 3.11 or newer.
- Linux or macOS for the calculator worker, which uses Unix resource limits.
- SQLite with FTS5 support (normally included with Python).
- Internet access for installation, arXiv discovery/downloads, and model requests.
- An OpenRouter API key for `ask` and `evaluate --live`. Other commands need no model credential.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
research-agent --help
research-agent config
```

For the pinned dependency snapshot instead of the version ranges in `pyproject.toml`:

```bash
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

The installed package is named `research_agent`; setuptools maps it to the source files in `src/`. `python -m research_agent` is also supported after installation. Run commands from the repository root, or supply an explicit configuration path.

## Quick start

Collect and index up to five papers about the topic in `config.yaml`, then search the corpus without using a model:

```bash
research-agent collect
research-agent papers
research-agent retrieve "small language models" --limit 3
```

To try a single known paper:

```bash
research-agent collect --query 'id:1706.03762' --limit 1
research-agent retrieve "attention"
```

To enable model answers, export your API key in your shell or `~/.bashrc`:

```bash
export OPENROUTER_API_KEY='your-key-here'
research-agent ask "What methods and limitations are reported in the indexed papers?" --local-only
```

By default, credentials in `~/.bashrc` take precedence over the process environment. Set `credentials.prefer_bashrc: false` in YAML to prefer the current shell's exported key. The application does not load `.env` files. Keep credentials out of YAML and version control.

`--local-only` disables literature discovery and paper acquisition for the agent; answering and figure inspection still send requests to the configured model API. The default answer file is `data/runs/answer.json`; use `ask --output PATH` to override it. Answers and execution details are also printed as JSON on stdout; tool logs go to stderr.

## Configuration

Edit `config.yaml` to select the topic, models, paths, retrieval settings, processing limits, and execution budgets. Unknown settings, duplicate YAML keys, and invalid limits are rejected.

| Setting | Purpose |
| --- | --- |
| `topic` | Research topic; when `query` is null, derives `all:"<topic>"`. |
| `query` | Optional explicit arXiv search expression. |
| `data` | Corpus directory; use separate directories for unrelated topics. |
| `paper_limit` | Default number of papers collected. |
| `model`, `vision_model` | Provider model identifiers for answers and page inspection. |
| `api.base_url` | OpenAI-compatible Responses endpoint base URL. |
| `max_turns`, `max_tools`, `max_tokens`, `seconds` | Agent execution budgets. |
| `processing` | PDF page, table, image, and rendering limits. |
| `retrieval` | Passage size and search limits. |
| `evaluation` | Evaluation input/output paths and live/local settings. |

Model identifiers are configurable; select models available to your account that support Responses API tool calls, and image input for page inspection. Provider availability, rate limits, and pricing can affect live runs.

Paths inside YAML resolve relative to the YAML file's directory. Explicit command-line paths resolve relative to the current working directory. Global options must come **before** the subcommand:

```bash
research-agent --config /path/to/config.yaml --data /path/to/corpus papers
```

`--data` changes the corpus location only; it does not relocate configured answer or evaluation output paths. Settings come from YAML followed by supported CLI overrides; arbitrary `RESEARCH_*` environment overrides are not implemented.

## Commands

| Command | Description |
| --- | --- |
| `config` | Validate and print effective non-secret configuration. |
| `collect [--query QUERY] [--limit N]` | Discover metadata, download PDFs, and index pages. |
| `collect --metadata-only` | Save metadata without downloading or indexing PDFs. |
| `collect --manifest FILE` | Replay metadata and verify downloaded PDFs against a pinned manifest. |
| `index [--force]` | Index PDFs for saved metadata; `--force` rebuilds existing page records. |
| `papers` | List saved metadata. |
| `retrieve QUERY [--limit N]` | Search local indexed passages without a model. |
| `ask QUESTION [--local-only] [--output FILE]` | Run the research agent. |
| `evaluate [--cases FILE] [--output FILE] [--live]` | Evaluate retrieval, optionally including live agent answers. |

Use `research-agent COMMAND --help` for all command options. Boolean overrides also accept `--no-...`, such as `--no-metadata-only` or `--no-local-only`.

`collect --manifest` expects JSON with a `papers` array. Each item contains `metadata` (the fields printed by `papers`) and `sha256` (the PDF's SHA-256 hex digest). No manifest generator is bundled.

## Evaluation

Create your own JSONL file with one case per line. No evaluation corpus or question set is bundled. Evidence must refer to papers and pages already indexed in your corpus:

```json
{"id":"attention-method","question":"What method is proposed?","retrieval_query":"attention","evidence":[{"paper_id":"1706.03762","page":1}],"required_tools":["paper_retrieval"],"unanswerable":false,"answer_terms":["attention"]}
```

```bash
research-agent evaluate --cases evaluation/questions.jsonl --output data/runs/evaluation.json
# Also run the model (requires credentials):
research-agent evaluate --cases evaluation/questions.jsonl --live
```

Retrieval evaluation reports page recall and precision. Live evaluation also reports citation provenance, expected citation recall, tool coverage, and lexical answer checks. These are proxy metrics; scientific correctness and evidence support require human review.

## Tests

```bash
python -m pytest -q
python -m pip check
```

Tests run without API credentials or internet access. They create synthetic PDFs in temporary directories and cover installed CLI entry points, PDF indexing/reindexing/rendering, retrieval, evaluation, calculator subprocesses, configuration errors, mocked arXiv discovery/download/cache, and an agent tool-call cycle with a fake model. Synthetic test results are not scientific findings.

Verification on Linux with Python 3.14.6: all seven tests passed, dependency checks passed, and a distributable wheel built successfully. A live smoke test downloaded and indexed all 15 pages of `1706.03762`, retrieved cited passages, and completed a model answer with validated citations. An earlier model answer was rejected by citation validation; live model output can vary.

## Files and outputs

```text
config.yaml          Validated application settings
pyproject.toml       Package metadata, dependencies, and CLI entry point
requirements.lock    Pinned dependency snapshot
src/                 research_agent package source
  cli.py             Commands and workflow orchestration
  agent.py           Model client, execution budgets, citation validation
  literature.py      Cached, rate-limited arXiv requests
  storage.py         SQLite metadata, pages, and full-text retrieval
  processing.py      PDF text, tables, images, and page rendering
  tools.py           Tool schemas, validation, execution, and traces
  calculator.py      Restricted numeric-expression worker
  evaluation.py      Retrieval and live-agent evaluation
  config.py          YAML validation and path resolution
  credentials.py     Credential lookup
  models.py          Metadata and tool input models
  logging.py         JSON log formatting
  __main__.py        python -m research_agent entry point
tests/               Offline regression and integration tests
```

The default `data/` directory contains `research.sqlite`, downloaded PDFs in `papers/`, extracted images in `assets/`, cached discovery results in `cache/`, and JSON reports in `runs/`. Generated data and the virtual environment are ignored by Git. The `diagnostics` and `examples` YAML sections are reserved settings; no diagnostic or example-capture command is bundled.

## Troubleshooting and limitations

- **`research_agent` cannot be imported:** activate `.venv` and reinstall with `python -m pip install -e '.[test]'`. Do not run `src/cli.py` directly or add `src/` to `PYTHONPATH`, where `logging.py` could shadow the standard library.
- **Configuration cannot be read:** run from the repository root or pass `--config /absolute/path/config.yaml`.
- **Empty retrieval:** metadata alone is not searchable PDF evidence. Run `collect` without `--metadata-only`; inspect its `ingestion` results for failures, then use `index` to retry processing downloaded files.
- **arXiv errors:** check connectivity and retry later. Requests are cached and paced at least three seconds apart; inaccessible PDFs can produce per-paper ingestion errors. Inspect JSON results even when a batch command exits successfully.
- **Model request fails:** check the credential, configured model, provider access, network, timeout, and budgets. Inspect `stop_reason` and `api_error` in the answer JSON; an answer report can contain a model error even when the CLI exits successfully.
- **Missing evaluation file:** create a JSONL case file and pass `evaluate --cases PATH`.
- **Scanned or complex PDFs:** text extraction does not perform OCR. Table extraction returns unverified candidates; source-page inspection can help when text or tables are incomplete.

Retrieval is lexical rather than embedding-based. Citation checks confirm page provenance, not whether a claim is scientifically supported by that page. The calculator accepts only restricted numeric expressions and supplied lists; it does not run arbitrary Python programs.
