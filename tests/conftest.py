import io
import random
from pathlib import Path

import pytest
import yaml
from PIL import Image

from shelfscan.schemas import CATEGORY_MAP


def fixture_jpeg(seed: int, size: tuple[int, int] = (64, 96)) -> bytes:
    rng = random.Random(seed)
    image = Image.new("RGB", size)
    # Coarse independent patterns give visibly distinct low-resolution fingerprints.
    for y in range(0, size[1], 8):
        for x in range(0, size[0], 8):
            color = tuple(rng.randrange(256) for _ in range(3))
            image.paste(color, (x, y, min(x + 8, size[0]), min(y + 8, size[1])))
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=90)
    return stream.getvalue()


@pytest.fixture
def dataset_config(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    source = tmp_path / "originals"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    for category_index, folder in enumerate(CATEGORY_MAP):
        directory = source / folder
        directory.mkdir(parents=True)
        for number in range(5):
            (directory / f"photo-{number}.jpg").write_bytes(
                fixture_jpeg(category_index * 100 + number)
            )
    config = {
        "source_root": "../originals",
        "expected_images": 30,
        "category_map": CATEGORY_MAP,
        "near_duplicate_distance": 0,
    }
    path = config_dir / "data.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path
