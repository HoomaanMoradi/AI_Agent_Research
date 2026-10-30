import re
import unicodedata
from pydantic import BaseModel, ConfigDict, Field


def stable_id(value: str) -> str:
    value = re.sub(r"^https?://(?:export\.)?arxiv.org/(?:abs|pdf)/", "", value)
    value = re.sub(r"\.pdf$", "", value)
    value = re.sub(r"v\d+$", "", value)
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-zA-Z-]+(?:\.[A-Z]{2})?/\d{7})", value):
        raise ValueError("Expected a valid arXiv identifier")
    return value


def normalized_title(title: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", title).casefold())


class Paper(BaseModel):
    id: str
    title: str
    authors: list[str]
    published: str
    abstract: str
    url: str
    pdf_url: str | None = None


class Inputs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Search(Inputs):
    query: str = Field(min_length=1, max_length=5000)
    limit: int | None = Field(default=None, ge=1, le=100)


class Acquire(Inputs):
    paper_id: str


class Retrieve(Inputs):
    query: str = Field(min_length=1, max_length=5000)
    limit: int | None = Field(default=None, ge=1, le=100)


class Figure(Inputs):
    paper_id: str
    page: int = Field(ge=1)
    question: str = Field(min_length=1, max_length=8000)


class Tables(Inputs):
    paper_id: str
    page: int = Field(ge=1)


class Calculate(Inputs):
    expression: str = Field(min_length=1, max_length=20000)
    data: dict[str, list[float]] = Field(default_factory=dict)


class ToolError(BaseModel):
    type: str
    message: str


class ToolResult(BaseModel):
    ok: bool
    result: dict | None = None
    error: ToolError | None = None
