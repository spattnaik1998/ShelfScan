import io

import pytest
from PIL import Image

from scripts.common import read_inventory
from scripts.prepare_images import prepare_dataset
from shelfscan.images import ImageInputError, prepare_image


def image_bytes(size=(200, 100), orientation=None):
    image = Image.new("RGB", size, "red")
    exif = Image.Exif()
    if orientation:
        exif[274] = orientation
    stream = io.BytesIO()
    image.save(stream, format="JPEG", exif=exif)
    return stream.getvalue()


def test_exif_orientation_applied_once_and_stripped():
    raw = image_bytes(orientation=6)
    output = prepare_image(raw)
    assert (output.source_width, output.source_height) == (200, 100)
    assert (output.width, output.height) == (100, 200)
    assert output.exif_orientation == 6
    with Image.open(io.BytesIO(output.jpeg)) as image:
        assert image.getexif().get(274) is None
    assert prepare_image(output.jpeg).width == 100


def test_resize_preserves_aspect_and_does_not_upscale():
    output = prepare_image(image_bytes((4096, 3072)))
    assert (output.width, output.height) == (1024, 768)
    assert len(output.jpeg) < 400 * 1024
    small = prepare_image(image_bytes((40, 80)))
    assert (small.width, small.height) == (40, 80)


@pytest.mark.parametrize(
    "raw,kwargs,code",
    [
        (b"not a photo", {}, "bad_image"),
        (b"x" * 20, {"max_bytes": 10}, "image_too_large"),
        (image_bytes(), {"max_pixels": 10}, "image_too_large"),
    ],
)
def test_invalid_uploads_fail_with_explicit_codes(raw, kwargs, code):
    with pytest.raises(ImageInputError) as error:
        prepare_image(raw, **kwargs)
    assert error.value.code == code


def test_truncated_jpeg_rejected():
    with pytest.raises(ImageInputError):
        prepare_image(image_bytes()[:100])


def test_preparation_accounts_for_bad_file_and_preserves_originals(dataset_config):
    project = dataset_config.parent.parent
    originals = project.parent / "originals"
    bad = originals / "Fiction" / "corrupt.jpg"
    bad.write_bytes(b"corrupt")
    snapshots = {str(p): p.read_bytes() for p in originals.rglob("*.jpg")}
    report = prepare_dataset(dataset_config)
    assert report["input_count"] == 31
    assert report["failed_count"] == 1
    assert report["prepared_count"] == 30
    assert not report["count_matches_expected"]
    assert all(p.read_bytes() == snapshots[str(p)] for p in originals.rglob("*.jpg"))
    rows = read_inventory(project / "data/inventory.jsonl")
    assert next(r for r in rows if r["file"].endswith("corrupt.jpg"))["error"] == "bad_image"
