import re
import time
import contextlib
import sys
import pymupdf
from .storage import Store
from .config import ProcessingSettings


class Processor:
    def __init__(self, store: Store, settings: ProcessingSettings | None = None):
        self.settings = settings or ProcessingSettings()
        self.store = store
        self.deadline = float("inf")

    def ingest(self, paper_id: str, force: bool = False) -> dict:
        paper = self.store.paper(paper_id)
        existing = self.store.db.execute("SELECT count(*) FROM pages WHERE paper_id=?", (paper.id,)).fetchone()[0]
        if existing and not force:
            return {"paper_id": paper.id, "pages": existing, "cached": True}
        pages = []
        section = "Unidentified section"
        with pymupdf.open(self.store.pdf_path(paper.id)) as doc:
            if len(doc) > self.settings.max_pages:
                raise ValueError(f"PDF exceeds {self.settings.max_pages}-page processing limit")
            for number, page in enumerate(doc, 1):
                if time.monotonic() >= self.deadline:
                    raise TimeoutError("Document processing deadline reached")
                text = page.get_text(sort=True)
                headings = re.findall(r"(?m)^[ \t]*(?:[1-9](?:\.\d{1,2})?)[ \t]+[A-Z][^\n]{2,85}$", text)
                if headings:
                    section = headings[0].strip()
                tables = []
                try:
                    with contextlib.redirect_stdout(sys.stderr):
                        detected = page.find_tables().tables if self.settings.extract_tables else []
                    for i, table in enumerate(detected):
                        tables.append({"id": f"{paper.id}:p{number}:t{i+1}", "bbox": list(table.bbox), "rows": table.extract(), "paper_id": paper.id, "page": number})
                except Exception:
                    tables = [{"error": "Table detection failed; inspect source page"}]
                # Embedded raster images; vector figures remain visible in source-page inspection.
                figures = []
                assets = self.store.root / "assets" / paper.id.replace("/", "_")
                for i, image in enumerate((page.get_images()[:self.settings.max_images_per_page] if self.settings.extract_images else [])):
                    info = {"xref": image[0], "width": image[2], "height": image[3]}
                    if self.settings.min_image_dimension <= image[2] <= self.settings.max_image_dimension and self.settings.min_image_dimension <= image[3] <= self.settings.max_image_dimension:
                        extracted = doc.extract_image(image[0])
                        if extracted and len(extracted["image"]) <= self.settings.max_image_bytes:
                            assets.mkdir(parents=True, exist_ok=True)
                            extension = extracted["ext"]
                            if re.fullmatch(r"[a-z0-9]+", extension):
                                target = assets / f"p{number}-image{i+1}.{extension}"
                                target.write_bytes(extracted["image"])
                                info["path"] = str(target.relative_to(self.store.root))
                    figures.append(info)
                captions = re.findall(r"(?im)^[ \t]*(?:fig(?:ure)?\.?|table)\s*\d+[^\n]*(?:\n[^\n]+)?", text)
                pages.append({"page": number, "section": section, "text": text, "tables": tables, "figures": {"images": figures, "captions": captions}})
        self.store.replace_pages(paper.id, pages)
        return {"paper_id": paper.id, "pages": len(pages), "empty_pages": [p["page"] for p in pages if len(p["text"].strip()) < self.settings.low_text_chars], "cached": False}

    def render(self, paper_id: str, page: int) -> bytes:
        self.store.page(paper_id, page)
        with pymupdf.open(self.store.pdf_path(paper_id)) as doc:
            source = doc[page-1]
            scale = min(self.settings.render_scale, self.settings.render_max_dimension / max(source.rect.width, source.rect.height))
            return source.get_pixmap(matrix=pymupdf.Matrix(scale, scale)).tobytes("png")
