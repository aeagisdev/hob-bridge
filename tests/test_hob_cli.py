"""Tests for tools/hob.py (hob sync) — a fake S3 client, no boto3 or network needed.

Run with pytest, or plain: python3 tests/test_hob_cli.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import hob  # noqa: E402


class FakeS3:
    def __init__(self, objects):
        self.objects = dict(objects)  # key -> size
        self.uploaded = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        fake = self

        class Paginator:
            def paginate(self, Bucket, Prefix):
                keys = sorted(k for k in fake.objects if k.startswith(Prefix))
                for start in range(0, len(keys), 2):  # several pages
                    yield {"Contents": [{"Key": k, "Size": fake.objects[k]} for k in keys[start:start + 2]]}

        return Paginator()

    def upload_file(self, filename, bucket, key, Callback=None, **kwargs):
        size = os.path.getsize(filename)
        if Callback:
            Callback(size)
        self.uploaded.append(key)
        self.objects[key] = size


def _tree(files):
    root = tempfile.mkdtemp()
    for rel, size in files.items():
        path = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"x" * size)
    return root


LOCAL = {"Stable-diffusion/a.safetensors": 10, "Lora/new.safetensors": 3, "Lora/changed.safetensors": 5,
         ".DS_Store": 1, "Lora/partial.safetensors.part": 2}
REMOTE = {"models/Stable-diffusion/a.safetensors": 10, "models/Lora/changed.safetensors": 4,
          "models/VAE/old.pt": 7, "models/": 0}


def test_scan_local_skips_dotfiles_and_partials():
    root = _tree(LOCAL)
    assert hob.scan_local(root) == {"Stable-diffusion/a.safetensors": 10, "Lora/new.safetensors": 3,
                                    "Lora/changed.safetensors": 5}


def test_plan():
    root = _tree(LOCAL)
    remote = hob.scan_remote(FakeS3(REMOTE), "vol", "models")
    assert remote == {"Stable-diffusion/a.safetensors": 10, "Lora/changed.safetensors": 4, "VAE/old.pt": 7}
    new, changed, remote_only = hob.plan(hob.scan_local(root), remote)
    assert new == ["Lora/new.safetensors"]
    assert changed == ["Lora/changed.safetensors"]
    assert remote_only == ["VAE/old.pt"]


def test_sync_modes():
    root = _tree(LOCAL)
    dry = FakeS3(REMOTE)
    assert hob.sync(dry, "vol", root, dry_run=True) == 0
    assert dry.uploaded == []

    only_new = FakeS3(REMOTE)
    hob.sync(only_new, "vol", root, only_new=True)
    assert only_new.uploaded == ["models/Lora/new.safetensors"]

    full = FakeS3(REMOTE)
    hob.sync(full, "vol", root)
    assert full.uploaded == ["models/Lora/new.safetensors", "models/Lora/changed.safetensors"]
    assert "models/VAE/old.pt" in full.objects  # never deleted
    hob.sync(full, "vol", root)  # second run: in sync
    assert len(full.uploaded) == 2


def test_scan_remote_retries_flaky_listing():
    class Flaky(FakeS3):
        calls = 0

        def get_paginator(self, name):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("The same next token was received twice")
            return super().get_paginator(name)

    assert hob.scan_remote(Flaky(REMOTE), "vol", "models", pause=0)["VAE/old.pt"] == 7


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
