"""
Main preprocessing entry point.
Routes input files (.pptx or .pdf) to the right extractor and
writes the normalized markdown to disk.
"""

from pathlib import Path
from typing import Optional

try:
    from .pptx_extractor import PPTXExtractor
    from .pdf_extractor import PDFExtractor
except ImportError:
    from pptx_extractor import PPTXExtractor
    from pdf_extractor import PDFExtractor

class Preprocessor:
    def __init__(self, use_florence: bool = True, lecture_num: int = 0):
        self.florence = None
        if use_florence:
            # Lazy import: only require torch/transformers if images are on
            try:
                from .florence_processor import FlorenceProcessor
            except ImportError:
                from florence_processor import FlorenceProcessor
            self.florence = FlorenceProcessor()
        self.lecture_num = lecture_num

    def process(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Returns the normalized markdown.
        If output_path is given, also writes it to disk.
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(input_path)

        ext = path.suffix.lower()
        if ext == ".pptx":
            extractor = PPTXExtractor(self.florence, self.lecture_num)
        elif ext == ".pdf":
            extractor = PDFExtractor(self.florence, self.lecture_num)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        markdown = extractor.extract(str(path))

        if output_path:
            Path(output_path).write_text(markdown, encoding="utf-8")

        return markdown


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CESS preprocessing pipeline")
    parser.add_argument("input", help="Path to .pptx or .pdf")
    parser.add_argument("-o", "--output", help="Output markdown path")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip Florence-2 image processing")
    parser.add_argument("--lecture", type=int, default=0,
                        help="Lecture number for citation metadata")
    args = parser.parse_args()

    pre = Preprocessor(use_florence=not args.no_images,
                       lecture_num=args.lecture)
    md = pre.process(args.input, args.output)

    if not args.output:
        print(md)
