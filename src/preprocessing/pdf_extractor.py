"""
PDF extractor.
Per report 5.2.2: uses PyMuPDF (fitz) and applies heuristics for:
  - page-level slide segmentation (## Slide N)
  - heading detection via font size
  - bullet recognition via leading symbols
  - paragraph boundary inference
  - image extraction for Florence-2
"""

import io
import re
import hashlib  # Added for image deduplication
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF

try:
    from . import image_filter
except ImportError:
    import image_filter
    
BULLET_CHARS = (
    "•", "●", "○", "■", "□", "▪", "▫", "◦", "-", "*", "–", "—",
    "·",  # middle dot, common PDF fallback for •
    "▶", "►", "▸", "❖", "✓", "✗",
)


class PDFExtractor:
    def __init__(self, florence_processor=None, lecture_num=0):
        self.florence = florence_processor
        self.lecture_num = lecture_num
        # Track hashes of processed images to skip repeated logos/backgrounds
        self.seen_image_hashes = set()

    def extract(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        out = []
        total = len(doc)

        for page_idx, page in enumerate(doc, start=1):
            print(f"[PDF] Slide {page_idx}/{total}...", flush=True)
            out.append(self._extract_page(page, page_idx, doc))

        doc.close()
        return "\n\n".join(out)

    def _extract_page(self, page, slide_num: int, doc) -> str:
        # Compute the dominant body font size to use as the threshold for headings
        spans = self._collect_spans(page)
        body_size = self._estimate_body_size(spans)

        title = self._find_title(spans, body_size)
        header = f"## Slide {slide_num}: {title}" if title else f"## Slide {slide_num}"
        parts = [header]

        text_block = self._spans_to_markdown(spans, body_size, skip_title=title)
        if text_block:
            parts.append(text_block)

        # Slides 1 & 2 are usually Title/Agenda. We don't process their images
        # for content, but we DO hash them as the "PPTX template" so that the
        # same recurring elements (university logo, footer dome, background
        # band, etc.) are auto-deduped on later slides.
        if self.florence:
            if slide_num <= 2:
                self._register_template_images(page, doc)
            else:
                img_text = self._extract_images(page, doc, slide_num)
                if img_text:
                    parts.append(img_text)

        return "\n\n".join(parts)

    def _register_template_images(self, page, doc) -> None:
        """Hash every image on a template-defining slide so it gets skipped
        as boilerplate on all later slides."""
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base = doc.extract_image(xref)
                h = hashlib.md5(base["image"]).hexdigest()
                self.seen_image_hashes.add(h)
                print(f"  [template] registered hash {h[:8]} from slide {page.number+1}", flush=True)
            except Exception:
                continue

    @staticmethod
    def _collect_spans(page):
        """Pull all spans with their font size and position."""
        spans = []
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type") != 0:  # 0 = text
                continue
            for line in block.get("lines", []):
                line_text = ""
                max_size = 0
                font = ""
                bbox = line.get("bbox", [0, 0, 0, 0])
                for sp in line.get("spans", []):
                    line_text += sp.get("text", "")
                    if sp.get("size", 0) > max_size:
                        max_size = sp.get("size", 0)
                        font = sp.get("font", "")
                line_text = line_text.strip()
                if line_text:
                    spans.append({
                        "text": line_text,
                        "size": round(max_size, 1),
                        "y": bbox[1],
                        "x": bbox[0],
                        "font": font,
                    })
        return spans

    @staticmethod
    def _estimate_body_size(spans):
        if not spans:
            return 12.0
        sizes = {}
        for s in spans:
            sizes[s["size"]] = sizes.get(s["size"], 0) + 1
        return max(sizes.items(), key=lambda x: x[1])[0]

    @staticmethod
    def _find_title(spans, body_size):
        for s in spans:
            if s["size"] > body_size * 1.2 and len(s["text"]) < 120:
                return s["text"]
        return ""

    def _spans_to_markdown(self, spans, body_size, skip_title="") -> str:
        lines = []
        skipped_title = False

        for s in spans:
            text = s["text"]
            stripped_check = text.lstrip()
            if stripped_check.startswith("#") and not stripped_check.startswith("##"):
                continue
            if not skipped_title and skip_title and text == skip_title:
                skipped_title = True
                continue

            stripped = text.lstrip()
            if stripped and stripped[0] in BULLET_CHARS:
                body = stripped[1:].lstrip()
                lines.append(f"- {body}")
                continue

            if s["size"] > body_size * 1.15 and len(text) < 100:
                lines.append(text)
                continue

            lines.append(text)

        merged = []
        buffer = []
        pending_bullet = False  # True when we've seen a bare "- " and need text
        for ln in lines:
            if ln.startswith("-"):
                # Flush any prose buffer first
                if buffer:
                    merged.append(" ".join(buffer))
                    buffer = []
                body = ln[1:].lstrip()
                if body:
                    # Normal bullet with content on the same line
                    merged.append(f"- {body}")
                    pending_bullet = False
                else:
                    # Bare "- " marker; the actual content is on the next line(s)
                    pending_bullet = True
            else:
                if pending_bullet:
                    # Attach this line as the body of the previous orphan bullet
                    merged.append(f"- {ln}")
                    pending_bullet = False
                else:
                    buffer.append(ln)
        if buffer:
            merged.append(" ".join(buffer))

        return "\n".join(merged)

    def _extract_images(self, page, doc, slide_num: int) -> str:
        results = []
        img_count = 0
        page_height = page.rect.height

        for img_info in page.get_images(full=True):
            img_count += 1
            xref = img_info[0]
            
            # --- COORDINATE FILTERING (Header/Footer) ---
            # Converted PPTXs put irrelevant logos/page numbers in the margins.
            rects = page.get_image_rects(xref)
            if rects:
                r = rects[0]
                # Skip if image is in the top 8% or bottom 8% of the page
                if r.y1 < (page_height * 0.08) or r.y0 > (page_height * 0.92):
                    print(f"  [slide {slide_num}, img {img_count}] skipped: in header/footer band", flush=True)
                    continue

            try:
                base = doc.extract_image(xref)
                img_bytes = base["image"]

                # --- DEDUPLICATION (Hashing) ---
                # Check if we have seen this exact image data before
                h = hashlib.md5(img_bytes).hexdigest()
                if h in self.seen_image_hashes:
                    print(f"  [slide {slide_num}, img {img_count}] skipped: dedup hash {h[:8]} (template or repeat)", flush=True)
                    continue
                self.seen_image_hashes.add(h)

                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                
                # Filter by size/aspect ratio
                if image_filter.is_decoration_image(img):
                    print(f"  [slide {slide_num}, img {img_count}] skipped: size/aspect filter ({img.width}x{img.height})", flush=True)
                    continue

                text = self.florence.process_image(
                    img, slide_num=slide_num, img_num=img_count,
                    lecture_num=self.lecture_num
                )
                if not text:
                    continue
                results.append(text)
                
            except Exception as e:
                print(f"  [slide {slide_num}, img {img_count}] skipped: exception ({e})", flush=True)
                continue

        return "\n".join(results)