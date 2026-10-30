import json
import re
import sqlite3
from pathlib import Path
from .config import RetrievalSettings
from .models import Paper, normalized_title, stable_id


class Store:
    def __init__(self, root: Path, settings: RetrievalSettings | None = None):
        self.settings = settings or RetrievalSettings()
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "research.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS papers(id TEXT PRIMARY KEY, title_key TEXT UNIQUE, metadata TEXT);
        CREATE TABLE IF NOT EXISTS pages(paper_id TEXT, page INTEGER, section TEXT, text TEXT,
          tables_json TEXT, figures_json TEXT, PRIMARY KEY(paper_id,page));
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(paper_id UNINDEXED, page UNINDEXED,
          section, text, tokenize='porter unicode61');
        ''')

    def save(self, paper: Paper) -> str:
        paper = paper.model_copy(update={"id": stable_id(paper.id)})
        row = self.db.execute("SELECT id FROM papers WHERE id=? OR title_key=?",
                              (paper.id, normalized_title(paper.title))).fetchone()
        if row:
            return row["id"]
        with self.db:
            self.db.execute("INSERT INTO papers VALUES(?,?,?)", (paper.id, normalized_title(paper.title), paper.model_dump_json()))
        return paper.id

    def paper(self, paper_id: str) -> Paper:
        row = self.db.execute("SELECT metadata FROM papers WHERE id=?", (stable_id(paper_id),)).fetchone()
        if not row:
            raise ValueError("Paper not found; run literature_search first")
        return Paper.model_validate_json(row[0])

    def papers(self) -> list[Paper]:
        return [Paper.model_validate_json(r[0]) for r in self.db.execute("SELECT metadata FROM papers ORDER BY id")]

    def pdf_path(self, paper_id: str) -> Path:
        return self.root / "papers" / (stable_id(paper_id).replace("/", "_") + ".pdf")

    def page(self, paper_id: str, page: int) -> dict:
        row = self.db.execute("SELECT * FROM pages WHERE paper_id=? AND page=?", (stable_id(paper_id), page)).fetchone()
        if row is None:
            raise ValueError("Page unavailable; acquire and index this paper first")
        return dict(row)

    def search(self, query: str, limit: int | None = None) -> list[dict]:
        limit = self.settings.limit if limit is None else limit
        words = re.findall(r"\w+", query)[:self.settings.max_query_terms]
        if not words:
            return []
        match = " OR ".join('"' + word + '"' for word in words)
        rows = self.db.execute("SELECT paper_id,page,section,text,bm25(chunks) AS rank FROM chunks WHERE chunks MATCH ? ORDER BY rank LIMIT ?", (match, limit))
        return [dict(r) | {"citation": f'[{r["paper_id"]} p.{r["page"]}]',
                           "url": self.paper(r["paper_id"]).url, "title": self.paper(r["paper_id"]).title} for r in rows]

    def replace_pages(self, paper_id: str, pages: list[dict]) -> None:
        with self.db:
            self.db.execute("DELETE FROM pages WHERE paper_id=?", (paper_id,))
            self.db.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
            for p in pages:
                self.db.execute("INSERT INTO pages VALUES(?,?,?,?,?,?)", (paper_id, p["page"], p["section"], p["text"], json.dumps(p["tables"]), json.dumps(p["figures"])))
                # Paragraph boundaries when possible; chunks never cross page boundaries.
                paragraphs = re.split(r"\n\s*\n", p["text"])
                buffer = ""
                for para in paragraphs:
                    for start in range(0, len(para), self.settings.paragraph_chars):
                        part = para[start:start+self.settings.paragraph_chars]
                        if len(buffer) + len(part) + 2 > self.settings.chunk_chars:
                            self.db.execute("INSERT INTO chunks VALUES(?,?,?,?)", (paper_id, p["page"], p["section"], buffer))
                            buffer = ""
                        buffer += part + "\n\n"
                if buffer.strip():
                    self.db.execute("INSERT INTO chunks VALUES(?,?,?,?)", (paper_id, p["page"], p["section"], buffer.strip()))
