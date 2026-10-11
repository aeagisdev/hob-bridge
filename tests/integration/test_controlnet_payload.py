"""Integration: Hob Bridge's ControlNet payload round-trips through the real
ControlNetUnit model of the commit pinned in worker/Dockerfile.

Local UI-shaped unit -> _controlnet_unit_to_api -> JSON -> ControlNetUnit.from_dict
must give the same RGBA inputs and the same field values. The model's three source
files are downloaded at the pinned commit; torch is stubbed (annotations only) and
the base64 decoder replicates A1111's images.read.

    pip install "pydantic<2" numpy pillow opencv-python-headless requests pytest
    pytest tests/integration/test_controlnet_payload.py

Re-run when bumping the ControlNet commit in worker/Dockerfile.
"""
import base64
import io
import json
import os
import re
import sys
import types
import urllib.request

import pytest

pydantic = pytest.importorskip("pydantic")
if not pydantic.VERSION.startswith("1."):
    pytest.skip("the pinned ControlNet uses pydantic v1 (A1111 1.10 ships it)", allow_module_level=True)
np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
from PIL import Image, ImageOps  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FILES = ("internal_controlnet/args.py", "scripts/enums.py", "annotator/util.py")


def _pinned_commit():
    with open(os.path.join(ROOT, "worker", "Dockerfile")) as fh:
        return re.search(r"origin ([0-9a-f]{40})", fh.read()).group(1)


@pytest.fixture(scope="module")
def cn(tmp_path_factory):
    commit = _pinned_commit()
    pkg = tmp_path_factory.mktemp("controlnet")
    for rel in FILES:
        url = f"https://raw.githubusercontent.com/Mikubill/sd-webui-controlnet/{commit}/{rel}"
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_bytes(urllib.request.urlopen(url, timeout=60).read())
    sys.modules.setdefault("torch", types.SimpleNamespace(Tensor=object))
    sys.path[:0] = [str(pkg), os.path.join(ROOT, "scripts")]
    from internal_controlnet.args import ControlNetUnit
    from scripts import enums
    import hob_bridge

    slider = types.SimpleNamespace(minimum=-1, maximum=4096, value=512)
    ControlNetUnit.cls_match_module = staticmethod(lambda value: True)
    ControlNetUnit.cls_match_model = staticmethod(lambda value: True)
    ControlNetUnit.cls_get_preprocessor = staticmethod(lambda module: types.SimpleNamespace(
        slider_resolution=slider, slider_1=slider, slider_2=slider, tags=[]))

    def a1111_decode(encoding):  # external_code.to_base64_nparray -> api.decode_base64_to_image
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(encoding))))
        if img.mode in ("RGB", "P") and isinstance(img.info.get("transparency"), bytes):
            img = img.convert("RGBA")
        return np.array(img).astype("uint8")

    ControlNetUnit.cls_decode_base64 = staticmethod(a1111_decode)
    return types.SimpleNamespace(Unit=ControlNetUnit, enums=enums, to_api=hob_bridge._controlnet_unit_to_api)


def _arrays():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)
    rgba = rng.integers(0, 256, (16, 24, 4), dtype=np.uint8)
    stroke = np.zeros((16, 24, 4), dtype=np.uint8)
    stroke[4:8, 4:12] = 255
    return rgb, rgba, stroke


def _cases(enums):
    rgb, rgba, stroke = _arrays()
    return {
        "sketch pair, RGBA mask": dict(image={"image": rgb, "mask": stroke}),
        "sketch pair, RGBA image": dict(image={"image": rgba, "mask": np.zeros_like(stroke)}),
        "image + uploaded mask": dict(image=rgb, mask=stroke),
        "merge multi-input": dict(image=[{"image": rgb}, {"image": rgba}],
                                  input_mode=enums.InputMode.MERGE),
        "non-default enums and flags": dict(
            image=rgb, resize_mode=enums.ResizeMode.OUTER_FIT, control_mode=enums.ControlMode.CONTROL,
            hr_option=enums.HiResFixOption.LOW_RES_ONLY, save_detected_map=False, pixel_perfect=True,
            low_vram=True, weight=0.55, guidance_start=0.1, guidance_end=0.8,
            effective_region_mask=rgb, advanced_weighting=[0.5] * 13),
        "no image (img2img fallback)": dict(),
    }


CASE_NAMES = ["sketch pair, RGBA mask", "sketch pair, RGBA image", "image + uploaded mask",
              "merge multi-input", "non-default enums and flags", "no image (img2img fallback)"]


@pytest.mark.parametrize("name", CASE_NAMES)
def test_unit_round_trip(cn, name):
    local = cn.Unit(enabled=True, module="canny", model="control_v11p_sd15_canny",
                    **_cases(cn.enums)[name])
    remote = cn.Unit.from_dict(json.loads(json.dumps(cn.to_api(local))))

    local_rgba, remote_rgba = local.get_input_images_rgba(), remote.get_input_images_rgba()
    if local_rgba is None:
        assert remote_rgba is None
    else:
        assert len(local_rgba) == len(remote_rgba)
        assert all(np.array_equal(a, b) for a, b in zip(local_rgba, remote_rgba))

    images = ("image", "mask", "effective_region_mask", "is_ui")
    diffs = [key for key in cn.Unit.__fields__
             if key not in images and getattr(local, key) != getattr(remote, key)]
    assert diffs == []
    if local.effective_region_mask is not None:
        assert np.array_equal(local.effective_region_mask, remote.effective_region_mask)
