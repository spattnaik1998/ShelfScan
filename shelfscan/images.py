"""Bounded image decoding, shared offline/app preprocessing and visual fingerprints."""

import hashlib
import io
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError


class ImageInputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PreparedImage:
    jpeg: bytes
    source_width: int
    source_height: int
    exif_orientation: int | None
    width: int
    height: int
    pixel_sha256: str
    dhash: str


def difference_hash(image: Image.Image) -> str:
    """64-bit dHash; similarity is only provisional book-identity evidence."""
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = [small.getpixel((x, y)) for y in range(8) for x in range(9)]
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return f"{bits:016x}"


def prepare_image(
    raw: bytes,
    *,
    long_edge: int = 1024,
    quality: int = 85,
    max_bytes: int = 15 * 1024 * 1024,
    max_pixels: int = 40_000_000,
) -> PreparedImage:
    if not (1 <= long_edge <= 4096 and 1 <= quality <= 95):
        raise ValueError("Invalid image preparation settings")
    if len(raw) > max_bytes:
        raise ImageInputError("image_too_large", "Raw image exceeds upload limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                width, height = source.size
                if width * height > max_pixels:
                    raise ImageInputError("image_too_large", "Decoded image exceeds pixel limit")
                if getattr(source, "n_frames", 1) > 1:
                    raise ImageInputError("bad_image", "Animated or multi-frame images unsupported")
                orientation = source.getexif().get(274)
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
                pixel_hash = hashlib.sha256(
                    f"{image.width}x{image.height}:".encode() + image.tobytes()
                ).hexdigest()
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=quality, optimize=True)
                return PreparedImage(
                    output.getvalue(),
                    width,
                    height,
                    orientation,
                    image.width,
                    image.height,
                    pixel_hash,
                    difference_hash(image),
                )
    except ImageInputError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageInputError("image_too_large", "Unsafe decoded image size") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageInputError("bad_image", "Cannot decode image") from exc
