"""Validated YAML settings."""
from pathlib import Path
import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
import yaml


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True, allow_inf_nan=False)


class DiscoverySettings(Settings):
    request_interval_seconds: float = Field(default=3.1, ge=3)
    timeout_seconds: float = Field(default=40, gt=0)
    attempts: int = Field(default=4, ge=1)
    retry_backoff_base: float = Field(default=2, ge=1)
    max_retry_after_seconds: float = Field(default=60, gt=0)
    cache_ttl_seconds: float = Field(default=86400, ge=0)
    page_size: int = Field(default=10, ge=1, le=2000)
    scan_multiplier: int = Field(default=3, ge=1)
    max_metadata_bytes: int = Field(default=5_000_000, ge=1)
    max_pdf_bytes: int = Field(default=50_000_000, ge=1)
    tool_limit: int = Field(default=5, ge=1, le=100)
    max_tool_limit: int = Field(default=20, ge=1, le=100)
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "relevance"
    sort_order: Literal["ascending", "descending"] = "descending"
    user_agent: str = Field(default="arxiv-research-agent/0.1 (personal research)", min_length=1)

    @model_validator(mode="after")
    def check_limits(self) -> Self:
        if self.tool_limit > self.max_tool_limit:
            raise ValueError("tool_limit must not exceed max_tool_limit")
        return self


class RetrievalSettings(Settings):
    limit: int = Field(default=5, ge=1, le=100)
    max_tool_limit: int = Field(default=10, ge=1, le=100)
    max_query_chars: int = Field(default=500, ge=1, le=5000)
    max_query_terms: int = Field(default=40, ge=1)
    chunk_chars: int = Field(default=2000, ge=3)
    paragraph_chars: int = Field(default=1800, ge=1)

    @model_validator(mode="after")
    def check_limits(self) -> Self:
        if self.paragraph_chars + 2 > self.chunk_chars:
            raise ValueError("paragraph_chars + 2 must not exceed chunk_chars")
        if self.limit > self.max_tool_limit:
            raise ValueError("limit must not exceed max_tool_limit")
        return self


class ProcessingSettings(Settings):
    max_pages: int = Field(default=300, ge=1)
    low_text_chars: int = Field(default=40, ge=0)
    extract_tables: bool = True
    extract_images: bool = True
    max_images_per_page: int = Field(default=30, ge=0)
    min_image_dimension: int = Field(default=32, ge=1)
    max_image_dimension: int = Field(default=6000, ge=1)
    max_image_bytes: int = Field(default=5_000_000, ge=1)
    render_scale: float = Field(default=1.4, gt=0)
    render_max_dimension: int = Field(default=1600, ge=1)
    table_text_chars: int = Field(default=16000, ge=1)

    @model_validator(mode="after")
    def check_dimensions(self) -> Self:
        if self.min_image_dimension > self.max_image_dimension:
            raise ValueError("min_image_dimension must not exceed max_image_dimension")
        return self


class APISettings(Settings):
    base_url: str = "https://openrouter.ai/api/v1"
    max_retries: int = Field(default=0, ge=0)
    timeout_seconds: float = Field(default=60, gt=0)
    vision_timeout_seconds: float = Field(default=40, gt=0)
    vision_output_tokens: int = Field(default=1000, ge=128)
    vision_input_reserve_tokens: int = Field(default=6000, ge=1)
    max_vision_question_chars: int = Field(default=1000, ge=1, le=8000)

    @model_validator(mode="after")
    def check_url(self) -> Self:
        from urllib.parse import urlparse
        url = urlparse(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("base_url must be an HTTP(S) URL without embedded credentials")
        return self


class CredentialSettings(Settings):
    env_var: str = "OPENROUTER_API_KEY"
    bashrc: Path = Path("~/.bashrc")
    bash_executable: str = "bash"
    timeout_seconds: float = Field(default=5, gt=0)
    prefer_bashrc: bool = True

    @model_validator(mode="after")
    def check_variable(self) -> Self:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.env_var):
            raise ValueError("env_var must be a valid shell variable name")
        return self


class CalculatorSettings(Settings):
    timeout_seconds: float = Field(default=3, gt=0)
    cpu_seconds: int = Field(default=2, ge=1)
    memory_mb: int = Field(default=256, ge=32)
    open_files: int = Field(default=16, ge=8)
    max_expression_chars: int = Field(default=2000, ge=1, le=20000)
    max_input_bytes: int = Field(default=100000, ge=1000)
    max_ast_nodes: int = Field(default=200, ge=1, le=1000)
    max_list_items: int = Field(default=1000, ge=1)
    max_abs_number: float = Field(default=1e100, gt=0, allow_inf_nan=False)


class EvaluationSettings(Settings):
    cases: Path = Path("evaluation/questions.jsonl")
    output: Path = Path("data/runs/evaluation.json")
    retrieval_k: int = Field(default=5, ge=1, le=100)
    local_only: bool = True
    live: bool = False
    manifest_output: Path = Path("data/runs/corpus.json")
    examples_output: Path = Path("data/runs/examples.json")


class DiagnosticSettings(Settings):
    output: Path = Path("data/runs/api-check.json")
    query: str | None = None
    output_tokens: int = Field(default=1000, ge=128)
    timeout_seconds: float = Field(default=60, gt=0)


class LoggingSettings(Settings):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    http_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"


class ExampleSettings(Settings):
    question: str = Field(min_length=1)
    tool: Literal["literature_search", "paper_acquisition", "paper_retrieval", "figure_inspection", "table_access", "python_analysis"]
    arguments: dict


class Config(Settings):
    topic: str = Field(default="small language models", min_length=1)
    query: str | None = None
    data: Path = Path("data")
    paper_limit: int = Field(default=5, ge=1, le=100)
    model: str = Field(default="nvidia/nemotron-3.5-super-vl:free", min_length=1)
    vision_model: str = Field(default="nvidia/nemotron-3.5-super-vl:free", min_length=1)
    max_turns: int = Field(default=10, ge=1)
    max_tools: int = Field(default=16, ge=1)
    max_tokens: int = Field(default=24000, ge=128)
    output_tokens: int = Field(default=1800, ge=128)
    seconds: float = Field(default=180, gt=0)
    max_question_chars: int = Field(default=8000, ge=1)
    min_request_tokens: int = Field(default=128, ge=128)
    history_item_reserve_tokens: int = Field(default=512, ge=0)
    local_only: bool = False
    metadata_only: bool = False
    force_reindex: bool = False
    answer_output: Path | None = None
    input_price: float = Field(default=0, ge=0)
    output_price: float = Field(default=0, ge=0)
    api: APISettings = Field(default_factory=APISettings)
    credentials: CredentialSettings = Field(default_factory=CredentialSettings)
    discovery: DiscoverySettings = Field(default_factory=DiscoverySettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    calculator: CalculatorSettings = Field(default_factory=CalculatorSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    diagnostics: DiagnosticSettings = Field(default_factory=DiagnosticSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    examples: list[ExampleSettings] = Field(default_factory=list)

    @model_validator(mode="after")
    def derive_query(self) -> Self:
        if not self.topic.strip():
            raise ValueError("topic must not be blank")
        if self.query is None:
            object.__setattr__(self, "query", 'all:"' + self.topic.replace('"', " ") + '"')
        elif not self.query.strip():
            raise ValueError("query must be null (automatic) or non-empty")
        return self

    @classmethod
    def from_yaml(cls, path: Path | str = Path("config.yaml")) -> "Config":
        source = Path(path).expanduser().resolve()
        try:
            raw = yaml.load(source.read_text(), Loader=UniqueSafeLoader)
        except OSError as exc:
            raise ValueError(f"Cannot read configuration: {source}") from exc
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            location = f" at line {mark.line + 1}" if mark else ""
            raise ValueError(f"Invalid YAML configuration{location}: {source}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a YAML mapping of setting names")
        try:
            config = cls.model_validate(raw)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors(include_input=False))
            raise ValueError(f"Invalid configuration: {problems}") from exc
        # Predictable paths even when a profile is selected from another directory.
        def resolve(path: Path) -> Path:
            path = path.expanduser()
            return path.resolve() if path.is_absolute() else (source.parent / path).resolve()
        evaluation = config.evaluation.model_copy(update={name: resolve(getattr(config.evaluation, name)) for name in ("cases", "output", "manifest_output", "examples_output")})
        credentials = config.credentials.model_copy(update={"bashrc": resolve(config.credentials.bashrc)})
        diagnostics = config.diagnostics.model_copy(update={"output": resolve(config.diagnostics.output)})
        return config.model_copy(update={"data": resolve(config.data), "evaluation": evaluation, "credentials": credentials, "diagnostics": diagnostics, "answer_output": resolve(config.answer_output) if config.answer_output else None})


class UniqueSafeLoader(yaml.SafeLoader):
    """Safe YAML without silently accepting duplicate or misspelled keys."""


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise yaml.constructor.ConstructorError(None, None, "Expected unique string keys", key_node.start_mark)
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)
