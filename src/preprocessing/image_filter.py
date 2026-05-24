"""
Shared image filtering used by both PDF and PPTX extractors.
Decides whether an image should be sent to Florence or skipped as decoration,
and whether Florence's output is hallucinated noise that should be dropped.
"""

from PIL import Image


# Size thresholds. The filter is now area-based rather than dimension-based:
# a 955x172 code-screenshot box (area ~164k) is legitimate content even though
# its height is under 250. A 100x30 logo (area 3k) still gets rejected.
# We also reject anything narrower or shorter than a hard "definitely a sliver"
# floor of 80px to catch true bullets / single-glyph images.
MIN_IMG_DIM = 80          # was MIN_IMG_WIDTH/HEIGHT = 250; lowered to a hard floor
MIN_IMG_AREA = 50_000     # this stays; main signal for "is this content?"

# Extreme aspect ratios are decorative slivers or border strips.
MAX_ASPECT_RATIO = 8.0    # was 5.0; raised because wide code-screenshot boxes
                          #          like 1996x396 = ratio 5.0 are legitimate.

# Florence frequently emits these phrases when it hallucinates on tiny or
# cropped images (logos, single letters, decorations).
DECORATION_KEYWORDS = (
    "logo",
    "pendant",
    "watermark",
    "ampersand",
    "silhouette",
    "stark contrast",
    "white border",
    "blue circle",
    "blue background",
    "black background",
    "set against",
    "modern and professional",
    "trustworthiness",
    "the number",
    "in the shape of",
    "man's face",
    "woman's face",
    "person's face",
    "wearing a suit",
    "serious expression",
    "reference sheet",
    "the man in the image",
    "the woman in the image",
    "the person in the image",
    "black and white image of a",
    "black ink",
    "thin black border",
    "screenshot of a computer screen",
    "code written in",
    "a code in the middle",
    "list of different types of",
    "full moon",
    "night sky",
    "the moon is",
    "illuminated, casting",
    "bright light",
    "blue sky",
    "white clouds",
    "live camera",
)

# A smaller subset that is a near-certain decoration signal regardless of
# caption length (e.g. university logos, watermark stamps).
STRONG_DECORATION_KEYWORDS = (
    "logo",
    "watermark",
    "pendant",
    "silhouette",
    "blue circle",
    "set against",
    "trustworthiness",
    "stark contrast",
    "ampersand",
    "man's face",
    "woman's face",
    "person's face",
    "reference sheet",
)

# Common watermark / stock-photo / source URLs that show up as standalone text
WATERMARK_TEXTS = (
    "pixshark",
    "popsugar",
    "shutterstock",
    "istock",
    "getty",
    "alamy",
    "dreamstime",
    "123rf",
)


def is_decoration_image(img: Image.Image) -> bool:
    """Return True if an image is too small or oddly shaped to be content."""
    w, h = img.width, img.height
    # Hard floor: either dimension below 80px is definitely a sliver/bullet
    if w < MIN_IMG_DIM or h < MIN_IMG_DIM:
        return True
    # Main signal: total area must be substantial
    if w * h < MIN_IMG_AREA:
        return True
    # Extreme aspect ratios are decorative borders
    ratio = max(w / max(h, 1), h / max(w, 1))
    if ratio > MAX_ASPECT_RATIO:
        return True
    return False


def looks_like_noise_output(text: str) -> bool:
    """
    Drop Florence output that's almost certainly noise:
      - Empty or very short
      - OCR result that's a tiny number of short words (e.g. 'ASU', 'Lectures')
      - Pure watermark text like 'POPSUGAR', '@ www.pixshark.com'
      - Short-ish caption matching any decoration keyword
      - Any caption matching the strong-decoration keywords (regardless of length)
    """
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < 12:
        return True

    lower = stripped.lower()

    # Watermarks / stock sources, regardless of length
    for w in WATERMARK_TEXTS:
        if w in lower:
            return True

    # Strong decoration markers — drop no matter how long the caption is
    for kw in STRONG_DECORATION_KEYWORDS:
        if kw in lower:
            return True

    # Short Florence captions that read like generic decoration descriptions
    if len(stripped) < 120:
        for kw in DECORATION_KEYWORDS:
            if kw in lower:
                return True

    # OCR outputs that are just one or two short words (e.g. 'ASU', '113',
    # 'Lectures', '٢٠٠'). Real OCR'd content has multiple words and punctuation.
    words = stripped.split()
    if len(words) <= 2 and all(len(w) <= 12 for w in words):
        return True

    return False
