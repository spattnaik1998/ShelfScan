import json

import pytest

from annotation.cache import CorruptCache, RoleCache


def test_cache_keys_invalidate_when_model_revision_or_image_changes(tmp_path):
    original = {"model": "a", "revision": "v1", "image": "abc"}
    assert RoleCache.key(original) != RoleCache.key({**original, "revision": "v2"})
    assert RoleCache.key(original) != RoleCache.key({**original, "image": "def"})
    cache = RoleCache(tmp_path)
    key = cache.key(original)
    cache.save(key, {"provenance": original, "attempts": [{"status": "started"}], "result": None})
    assert cache.total_attempts() == 1
    assert cache.load(key)["result"] is None


def test_cache_corruption_fails_closed(tmp_path):
    cache = RoleCache(tmp_path)
    provenance = {"revision": "v1"}
    key = cache.key(provenance)
    cache.save(key, {"provenance": provenance, "attempts": [], "result": {"value": "safe"}})
    path = tmp_path / f"{key}.json"
    data = json.loads(path.read_bytes())
    data["payload"]["result"]["value"] = "corrupted"
    path.write_text(json.dumps(data))
    with pytest.raises(CorruptCache):
        cache.load(key)
