"""
Image processor: hybrid Tesseract + Florence-2.

  - Tesseract handles OCR (fast, accurate on printed text)
  - Florence-2 handles diagram captioning when OCR yields nothing meaningful

This is much faster than using Florence for both tasks, especially on CPU.
"""

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

import time  
try:
    from . import image_filter
    from . import tesseract_ocr
except ImportError:
    import image_filter
    import tesseract_ocr


class ImageProcessor:
    """
    Hybrid Processor: 
    - Uses Tesseract as a 'sensor' for prose (paragraphs).
    - Uses Florence-2 for everything else (diagrams, tables, photos).
    """

    def __init__(self, model_id="microsoft/Florence-2-base", device=None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = None
        self._processor = None

    def _ensure_florence(self):
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=self.dtype, trust_remote_code=True
            ).to(self.device)
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )

    def _florence_task(self, image: Image.Image, task_prompt: str) -> str:
        self._ensure_florence()
        inputs = self._processor(text=task_prompt, images=image, return_tensors="pt").to(self.device, self.dtype)
        generated_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(generated_text, task=task_prompt, image_size=image.size)
        return parsed.get(task_prompt, "")

    def caption(self, image: Image.Image) -> str:
        return self._florence_task(image, "<DETAILED_CAPTION>")

    def dense_caption(self, image: Image.Image) -> str:
        result = self._florence_task(image, "<DENSE_REGION_CAPTION>")
        if isinstance(result, dict):
            labels = result.get("labels", [])
            return "; ".join(labels) if labels else ""
        return str(result)

    def process_image(self, image: Image.Image, slide_num: int, img_num: int, lecture_num: int = 0) -> str:
        w, h = image.width, image.height
        print(f"  [slide {slide_num}, img {img_num}] processing {w}x{h}...", flush=True)

        # 1. Always run Tesseract first (cheap). Its output is used two ways:
        #    a) If it looks like a real prose page (Code-of-Ethics style),
        #       keep the OCR text plus a short Florence one-liner for context.
        #    b) Otherwise it just serves as a signal for classifying the
        #       image type (code? table? diagram?) below.
        t0 = time.time()
        ocr_text = tesseract_ocr.ocr_image(image)
        t_ocr = time.time() - t0

        if self._is_paragraph_prose(ocr_text):
            print(f"    -> Tesseract detected prose ({t_ocr:.1f}s). Adding short Florence context.", flush=True)
            short_caption = self._short_florence_caption(image)
            return self._format_prose_with_context(
                ocr_text, short_caption, slide_num, img_num, lecture_num
            )

        # 2. Non-prose image: use Florence to caption, then map that caption
        #    plus the Tesseract signal into one short domain-specific label.
        print(f"    -> Non-prose image. Classifying with Florence...", flush=True)
        t0 = time.time()
        florence_caption = self._short_florence_caption(image)
        t_flo = time.time() - t0

        if image_filter.looks_like_noise_output(florence_caption):
            # Florence produced obvious garbage. Fall back to a generic
            # label based on Tesseract signals alone.
            label = self._classify_image(ocr_text, "")
            if label is None:
                print(f"    -> Florence output filtered as noise; no fallback label.", flush=True)
                return ""
            print(f"    -> Florence noisy, using fallback label.", flush=True)
        else:
            label = self._classify_image(ocr_text, florence_caption)

        if label is None:
            print(f"    -> No useful label produced; dropping.", flush=True)
            return ""

        print(f"    -> Florence caption OK ({t_flo:.1f}s)  label='{label}'", flush=True)
        return self._format_label(label, slide_num, img_num, lecture_num)

    def _short_florence_caption(self, image: Image.Image) -> str:
        """One-line caption from Florence's short <CAPTION> task."""
        try:
            cap = self._florence_task(image, "<CAPTION>")
        except Exception:
            return ""
        cap = (cap or "").strip()
        # First sentence only, to keep things brief
        for end in (". ", "! ", "? "):
            if end in cap:
                cap = cap.split(end, 1)[0]
                break
        return cap

    # ---- Image classification into short, domain-specific labels --------

    # Markers in Tesseract output that strongly suggest source code.
    _CODE_MARKERS = (
        "import ", "from ", "def ", "class ", "print(", "return ",
        "self.", ">>>", "==", "!=", "->", "for ", "while ",
        "if ", "elif ", "else:", "lambda ", "yield ",
        "#include", "int ", "void ", "public ", "private ",
        "function", "var ", "let ", "const ",
    )

    # Words in the Florence caption that hint at a particular image type.
    _DIAGRAM_HINTS = (
        "diagram", "flowchart", "flow chart", "chart", "graph",
        "tree", "network", "architecture", "pipeline", "arrow",
        "boxes connected", "connected by arrows", "schema", "uml",
    )
    _TABLE_HINTS = (
        "table", "rows and columns", "spreadsheet", "grid",
    )
    _TERMINAL_HINTS = (
        "terminal", "console", "command line", "shell",
        "output", "black background with white text",
    )
    _PHOTO_HINTS = (
        "person", "people", "man", "woman", "book", "photo",
        "photograph", "scene", "sky", "landscape", "building",
    )

    @staticmethod
    def _classify_image(ocr_text: str, florence_caption: str):
        """Return a short domain-specific label for a non-prose image, or
        None if nothing meaningful could be inferred."""
        ocr_lower = (ocr_text or "").lower()
        cap_lower = (florence_caption or "").lower()

        # 1. Strong code signal from Tesseract -> always "A code snippet"
        code_hits = sum(1 for m in ImageProcessor._CODE_MARKERS if m in ocr_lower)
        code_chars = sum(1 for c in (ocr_text or "") if c in "()=:[]{}")
        if code_hits >= 2 and len(ocr_text or "") > 20 and code_chars >= 4:
            return "A code snippet"

        # 2. Caption hints
        if any(h in cap_lower for h in ImageProcessor._TABLE_HINTS):
            return "A comparison or reference table"
        if any(h in cap_lower for h in ImageProcessor._DIAGRAM_HINTS):
            return "A diagram or flowchart illustrating the concept"
        if any(h in cap_lower for h in ImageProcessor._TERMINAL_HINTS):
            return "A program output or terminal screenshot"

        # 3. OCR has a chunk of structured text but isn't code -> likely a table
        ocr_lines = [ln for ln in (ocr_text or "").splitlines() if ln.strip()]
        if len(ocr_lines) >= 4 and not code_hits:
            return "A comparison or reference table"

        # 4. Photo-like Florence caption
        if any(h in cap_lower for h in ImageProcessor._PHOTO_HINTS):
            # Decorative photos rarely add value; drop them.
            return None

        # 5. We got a real Florence caption but no specific type matched.
        #    Keep it but trim hard. Useful for genuine illustrations.
        if florence_caption and len(florence_caption) > 12:
            short = florence_caption.strip().rstrip(".")
            if len(short) > 90:
                short = short[:90].rsplit(" ", 1)[0]
            return f"An illustration: {short}"

        return None

    _FUNCTION_WORDS = frozenset((
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "be", "been", "being", "of", "in", "on", "at", "to", "for", "with",
        "by", "from", "as", "that", "this", "these", "those", "it", "its",
        "he", "she", "they", "we", "you", "i", "his", "her", "their", "our",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "can", "could", "must", "if", "then",
        "than", "so", "because", "while", "when", "where", "which", "who",
        "what", "whose", "all", "any", "some", "no", "not", "such",
        "into", "out", "over", "under", "between", "among", "through",
    ))

    @staticmethod
    def _is_paragraph_prose(text: str) -> bool:
        if not text:
            return False
        words = text.strip().split()
        
        if len(words) < 12: # Slightly lowered from 15
            return False

        if not any(c in text for c in ".,;:"):
            return False

        normalized = [w.lower().strip(",.;:!?()[]{}\"'") for w in words]
        fn_count = sum(1 for w in normalized if w in ImageProcessor._FUNCTION_WORDS)
        
        # Threshold tuned so that real prose pages pass and everything else
        # falls through to Florence:
        #   - Code of Ethics-style slide: ~38% -> pass
        #   - Plain paragraph slide:      ~30% -> pass
        #   - Stop-words list:            ~52% -> pass
        #   - Regex character table:      ~22% -> fail (sent to Florence)
        #   - Code snippet:                <5% -> fail (sent to Florence)
        #   - Diagram labels:              <12% -> fail (sent to Florence)
        if (fn_count / len(words)) < 0.25:
            return False

        # VERY IMPORTANT: Numbered lists (1., 2.) create many short tokens.
        # Raise this from 0.10 to 0.30
        short_tokens = sum(1 for w in words if len(w) <= 1)
        if short_tokens / len(words) > 0.30: 
            return False

        return True

    @staticmethod
    def _format_prose_with_context(ocr_text: str, short_caption: str,
                                   slide_num: int, img_num: int,
                                   lecture_num: int) -> str:
        """Used when an image is essentially a slide-screenshot of prose
        text (e.g. Code of Ethics). Keeps the OCR text and adds a brief
        Florence one-liner for context."""
        ctx = ""
        if short_caption and not image_filter.looks_like_noise_output(short_caption):
            ctx = f"\n_Context: {short_caption.strip().rstrip('.')}._\n"
        return (
            f"\n```\n{ocr_text}\n```\n"
            f"{ctx}"
            f"[Source: Lecture {lecture_num}, Slide {slide_num}, Image {img_num}]\n"
        )

    @staticmethod
    def _format_label(label: str, slide_num: int, img_num: int,
                      lecture_num: int) -> str:
        """Used for non-prose images: emit a short domain-specific label."""
        return (
            f"\n[Figure {slide_num}.{img_num}] {label}.\n"
            f"[Source: Lecture {lecture_num}, Slide {slide_num}]\n"
        )

# Backwards-compatible alias so existing imports keep working
FlorenceProcessor = ImageProcessor
