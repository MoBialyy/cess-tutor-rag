"""
PPTX extractor.
Per report 5.2.2: uses python-pptx to preserve slide boundaries,
text hierarchy (title/body/bullets), and extract embedded images.
Outputs normalized Markdown with ## Slide N / ### subheaders.
"""

import io
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.util import Emu

try:
    from . import image_filter
except ImportError:
    import image_filter


class PPTXExtractor:
    def __init__(self, florence_processor=None, lecture_num=0):
        self.florence = florence_processor
        self.lecture_num = lecture_num

    def extract(self, pptx_path: str) -> str:
        """Returns the full normalized markdown for the deck."""
        prs = Presentation(pptx_path)
        out = []
        total = len(prs.slides)

        for slide_idx, slide in enumerate(prs.slides, start=1):
            print(f"[PPTX] Slide {slide_idx}/{total}...", flush=True)
            out.append(self._extract_slide(slide, slide_idx))

        return "\n\n".join(out)

    def _extract_slide(self, slide, slide_num: int) -> str:
        title = self._get_slide_title(slide)
        header = f"## Slide {slide_num}: {title}" if title else f"## Slide {slide_num}"
        parts = [header]

        img_count = 0
        for shape in slide.shapes:
            # Skip the title shape, already used
            if shape.has_text_frame and shape == slide.shapes.title:
                continue

            if shape.has_text_frame:
                text_block = self._extract_text_frame(shape.text_frame)
                if text_block:
                    parts.append(text_block)

            elif shape.shape_type == 13 and self.florence and slide_num > 2:  # 13 = PICTURE
                img_count += 1
                img_text = self._extract_image(shape, slide_num, img_count)
                if img_text:
                    parts.append(img_text)

            elif shape.has_table:
                table_md = self._extract_table(shape.table)
                if table_md:
                    parts.append(table_md)

        return "\n\n".join(parts)

    @staticmethod
    def _get_slide_title(slide) -> str:
        if slide.shapes.title and slide.shapes.title.has_text_frame:
            return slide.shapes.title.text_frame.text.strip()
        return ""

    @staticmethod
    def _extract_text_frame(tf) -> str:
        """
        Convert a text frame's paragraphs to markdown.
        Bulleted paragraphs become '- item'; others become plain text.
        Level 0 paragraphs that look like headings (short, no period)
        become ### headers.
        """
        lines = []
        for para in tf.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # python-pptx exposes bullet via paragraph format; we use a
            # simple heuristic: if level > 0 OR para has a bullet, treat as list
            is_bullet = para.level > 0 or PPTXExtractor._has_bullet(para)
            if is_bullet:
                indent = "  " * para.level
                lines.append(f"{indent}- {text}")
            else:
                # heading-like? short, no terminal punctuation
                if len(text) < 80 and not text.endswith((".", "?", "!", ":")):
                    lines.append(f"### {text}")
                else:
                    lines.append(text)
        return "\n".join(lines)

    @staticmethod
    def _has_bullet(para) -> bool:
        # python-pptx doesn't expose bullets cleanly; check XML element
        pPr = para._pPr
        if pPr is None:
            return False
        # presence of buChar / buAutoNum indicates a bullet
        ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
        return (
            pPr.find(f"{ns}buChar") is not None
            or pPr.find(f"{ns}buAutoNum") is not None
        )

    def _extract_image(self, shape, slide_num: int, img_num: int) -> str:
        try:
            image_bytes = shape.image.blob
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Skip tiny / oddly-shaped images BEFORE running Florence
            if image_filter.is_decoration_image(img):
                return ""
            # process_image returns "" for noisy/decoration output internally
            return self.florence.process_image(
                img, slide_num=slide_num, img_num=img_num,
                lecture_num=self.lecture_num
            )
        except Exception as e:
            return f"\n[Image on slide {slide_num} could not be processed: {e}]\n"

    @staticmethod
    def _extract_table(table) -> str:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            rows.append("| " + " | ".join(cells) + " |")
        if not rows:
            return ""
        # markdown header separator after first row
        sep = "| " + " | ".join(["---"] * len(table.rows[0].cells)) + " |"
        return "\n".join([rows[0], sep] + rows[1:])