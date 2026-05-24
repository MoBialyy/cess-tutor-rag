"""
Constrained semantic chunker for normalized lecture markdown.

Per report section 4.3.7:
  - Target chunk size: 450-550 tokens (1 chunk ~= 3-4 slides)
  - Overlap: 75 tokens between adjacent chunks
  - Splits on a hierarchy of semantic boundaries: slide headers first,
    paragraphs next, sentences as fallback, words as last resort.
  - Token counting uses INSTRUCTOR-XL's tokenizer for accurate sizing
    against the embedding model's 512-token window.

Each chunk is emitted as a JSON line with:
  - chunk_id:    "<source_stem>_chunk_<N>"
  - source:      original file name (e.g. "Lecture_2_NLP.md")
  - slide_range: "5-8" or "12" depending on how many slides the chunk spans
  - topic:       title from the first slide in the chunk (best-effort)
  - tokens:      INSTRUCTOR-XL token count
  - text:        the chunk content
"""

import json
import re
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .token_counter import count_tokens
except ImportError:
    from token_counter import count_tokens


# Hierarchical separators (from report 4.3.7). Order matters: the splitter
# tries each separator in turn and only falls back to finer ones if the
# chunk still exceeds the size limit.
SEPARATORS = [
    "\n## ",      # Slide header  -- the most important pedagogical boundary
    "\n### ",     # Subsection header
    "\n#### ",    # Sub-subsection
    "\n\n",       # Paragraph break
    "\n",         # Line break
    ". ",         # Sentence boundary
    " ",          # Word boundary
    "",           # Character-level fallback
]

# Matches "## Slide 12: Some Title"  or  "## Slide 12"
SLIDE_HEADER_RE = re.compile(r"^##\s+Slide\s+(\d+)(?::\s*(.+))?\s*$", re.MULTILINE)


class Chunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Length is measured in INSTRUCTOR-XL tokens, not characters.
        # The splitter will call count_tokens() on candidate substrings.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=count_tokens,
            separators=SEPARATORS,
            is_separator_regex=False,
            keep_separator=True,
        )

    def chunk_text(self, text: str) -> List[str]:
        """Split a markdown string into chunks. Returns plain strings."""
        return self._splitter.split_text(text)

    def chunk_file(self, input_path: str, source_name: Optional[str] = None,
               lecture_num: Optional[int] = None,
               lecture_title: str = "") -> List[dict]:
        """
        Read a markdown file and return a list of chunk dicts ready for JSONL.

        source_name: used in chunk_id and metadata. Defaults to the file stem.
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(input_path)

        text = path.read_text(encoding="utf-8")
        stem = source_name or path.stem
        raw_chunks = self.chunk_text(text)

        results = []
        for i, chunk_text in enumerate(raw_chunks, start=1):
            slide_range, topic = self._extract_slide_info(chunk_text)
            results.append({
                "chunk_id": (f"lec{lecture_num}_{stem}_chunk_{i:03d}"
                             if lecture_num is not None
                             else f"{stem}_chunk_{i:03d}"),
                "source": path.name,
                "slide_range": slide_range,
                "topic": topic,
                "tokens": count_tokens(chunk_text),
                "text": chunk_text.strip(),
                "lecture_num": lecture_num,
                "lecture_title": lecture_title,
            })
        return results

    @staticmethod
    def _extract_slide_info(chunk_text: str):
        """
        Find the slide number(s) and topic referenced inside a chunk.
        Returns (slide_range_string, topic_string).

        Topic uses the first slide header in the chunk that *has* a title.
        Bare "## Slide N" headers (like cover pages) are skipped for the topic
        but still counted in the slide range.
        """
        matches = list(SLIDE_HEADER_RE.finditer(chunk_text))
        if not matches:
            return "unknown", ""

        slides = [int(m.group(1)) for m in matches]
        topic = ""
        for m in matches:
            t = (m.group(2) or "").strip()
            if t:
                topic = t
                break

        if len(slides) == 1:
            slide_range = str(slides[0])
        else:
            slide_range = f"{min(slides)}-{max(slides)}"

        return slide_range, topic

    @staticmethod
    def write_jsonl(chunks: List[dict], output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Chunk a normalized lecture markdown file into JSONL"
    )
    parser.add_argument("input", help="Path to .md file from preprocessing")
    parser.add_argument("-o", "--output", help="Output .jsonl path",
                        default=None)
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Target tokens per chunk (default 500)")
    parser.add_argument("--chunk-overlap", type=int, default=75,
                        help="Token overlap between chunks (default 75)")
    parser.add_argument("--source-name", default=None,
                        help="Override the source name used in chunk_id")
    args = parser.parse_args()

    chunker = Chunker(chunk_size=args.chunk_size,
                      chunk_overlap=args.chunk_overlap)

    print(f"Chunking {args.input}...", flush=True)
    chunks = chunker.chunk_file(args.input, source_name=args.source_name)
    print(f"  produced {len(chunks)} chunks", flush=True)

    # Quick size distribution summary
    sizes = [c["tokens"] for c in chunks]
    if sizes:
        print(f"  tokens: min={min(sizes)} max={max(sizes)} "
              f"avg={sum(sizes)//len(sizes)}", flush=True)

    out_path = args.output or str(Path(args.input).with_suffix(".jsonl"))
    chunker.write_jsonl(chunks, out_path)
    print(f"  wrote {out_path}", flush=True)
