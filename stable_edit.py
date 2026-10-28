"""Hard pixel boundaries for model-generated region and toggle edits.

This module validates a proposed region and copies ONLY candidate pixels inside
it. It draws no UI controls. A conservative connected-component check can also
reject switches with an unrecognizable track, thumb, or requested transition.
``verified`` in compositing evidence refers to the exterior pixel invariant.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import math

import numpy as np
from PIL import Image, UnidentifiedImageError


class StableEditError(ValueError):
    """Invalid or unverified local edit; retain the current image."""


def _dimensions(frame):
    if not isinstance(frame, dict):
        raise StableEditError("当前画面尺寸不正确，原图已保留。")
    width, height = frame.get("width"), frame.get("height")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (width, height)):
        raise StableEditError("当前画面尺寸不正确，原图已保留。")
    return width, height


def _bounds(value, width, height):
    if (not isinstance(value, list) or len(value) != 4
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)):
        raise StableEditError("开关区域必须是四个原生整数坐标，原图已保留。")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise StableEditError("开关区域超出原图或为空，原图已保留。")
    box_width, box_height = x1 - x0, y1 - y0
    if not (12 <= box_width <= 64 and 6 <= box_height <= 36):
        raise StableEditError("无法确定紧贴单个开关的区域，原图已保留。")
    if not 1.3 <= box_width / box_height <= 4.5:
        raise StableEditError("该区域不像单个开关，原图已保留。")
    if box_width * box_height > width * height * 0.03:
        raise StableEditError("开关区域过大，可能包含其他内容，原图已保留。")
    return [x0, y0, x1, y1]


def validate_toggle_plan(value, frame, event):
    """Return a canonical plan; bounds are native, half-open PNG coordinates.

    Geometry checks bound the damage, but cannot prove that an AI box excludes
    every label. Ambiguous recognition must be rejected by the planner too.
    """
    width, height = _dimensions(frame)
    if not isinstance(value, dict) or value.get("kind") != "toggle":
        raise StableEditError("本次点击未明确识别为单个开关，原图已保留。")
    bounds = _bounds(value.get("bounds"), width, height)
    current, target = value.get("current_state"), value.get("target_state")
    if current not in ("on", "off") or target not in ("on", "off") or current == target:
        raise StableEditError("开关前后状态不明确或没有翻转，原图已保留。")
    if not isinstance(event, dict) or event.get("type") != "click":
        raise StableEditError("局部开关编辑需要当前画面上的点击，原图已保留。")
    x, y = event.get("x"), event.get("y")
    if any(isinstance(item, bool) or not isinstance(item, (int, float))
           or (isinstance(item, float) and not math.isfinite(item))
           for item in (x, y)):
        raise StableEditError("点击坐标不正确，原图已保留。")
    x0, y0, x1, y1 = bounds
    if not (x0 <= x < x1 and y0 <= y < y1):
        raise StableEditError("点击没有落在已识别的开关内，原图已保留。")
    return {"kind": "toggle", "bounds": bounds,
            "current_state": current, "target_state": target}


def _region_bounds(value, width, height):
    if any(type(dimension) is not int or not 1 <= dimension <= 1024
           for dimension in (width, height)):
        raise StableEditError("局部编辑的原图尺寸须在 1–1024 像素以内，原图已保留。")
    if (not isinstance(value, list) or len(value) != 4
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)):
        raise StableEditError("局部区域必须是四个原生整数坐标，原图已保留。")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise StableEditError("局部区域超出原图或为空，原图已保留。")
    box_width, box_height = x1 - x0, y1 - y0
    if box_width < 2 or box_height < 2:
        raise StableEditError("局部区域至少需要 2×2 像素，原图已保留。")
    # Keep this integer-exact, including the 35% boundary. No floating point
    # inputs or rounding may enlarge the accepted region.
    if box_width * box_height * 100 > width * height * 35:
        raise StableEditError("局部区域超过原图的 35%，原图已保留。")
    return [x0, y0, x1, y1]


def validate_region_plan(value, frame, event):
    """Validate one generic click-local rectangle, without toggle semantics."""
    width, height = _dimensions(frame)
    if not isinstance(value, dict) or value.get("kind") != "region":
        raise StableEditError("本次操作没有明确的通用局部区域，原图已保留。")
    bounds = _region_bounds(value.get("bounds"), width, height)
    if not isinstance(event, dict) or event.get("type") != "click":
        raise StableEditError("局部编辑需要当前画面上的点击，原图已保留。")
    x, y = event.get("x"), event.get("y")
    if any(isinstance(item, bool) or not isinstance(item, (int, float))
           or (isinstance(item, float) and not math.isfinite(item))
           for item in (x, y)):
        raise StableEditError("点击坐标不正确，原图已保留。")
    x0, y0, x1, y1 = bounds
    if not (x0 <= x < x1 and y0 <= y < y1):
        raise StableEditError("点击没有落在已识别的局部区域内，原图已保留。")
    return {"kind": "region", "bounds": bounds}


def _decode_png(value):
    if not isinstance(value, bytes) or not value or len(value) > 8 * 1024**2:
        raise StableEditError("局部编辑图片无效，原图已保留。")
    try:
        with Image.open(BytesIO(value)) as image:
            if image.format != "PNG" or image.mode != "L" or getattr(image, "n_frames", 1) != 1:
                raise StableEditError("局部编辑需要同尺寸的单帧灰度 PNG，原图已保留。")
            if image.width * image.height > 1024**2:
                raise StableEditError("局部编辑图片尺寸过大，原图已保留。")
            image.load()
            return np.array(image, dtype=np.uint8, copy=True)
    except StableEditError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise StableEditError("局部编辑图片无法解码，原图已保留。") from None


def compose_toggle(original_png, candidate_png, bounds):
    """Return ``(PNG bytes, evidence)`` with an exact exterior pixel guarantee.

    Both inputs must already be identically sized L-mode PNGs. The candidate
    can differ everywhere: only the accepted rectangle is ever copied. No
    conversion, resizing, blending, feathering, or expansion is performed.
    """
    return _compose_locked(original_png, candidate_png, bounds,
        _bounds, "toggle", "locked_rectangle")


def compose_region(original_png, candidate_png, bounds):
    """Copy one <=35% rectangle, with no claim about its semantic content."""
    return _compose_locked(original_png, candidate_png, bounds,
        _region_bounds, "region", "local_region")


def _compose_locked(original_png, candidate_png, bounds, validate_bounds, kind, mode):
    """Shared lossless uint8 composition and final encoded-PNG verification."""
    original, candidate = _decode_png(original_png), _decode_png(candidate_png)
    if original.shape != candidate.shape:
        raise StableEditError("局部编辑的图片尺寸不一致，原图已保留。")
    height, width = original.shape
    x0, y0, x1, y1 = validate_bounds(bounds, width, height)
    mask = np.zeros(original.shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    result = original.copy()
    result[mask] = candidate[mask]
    changed = int(np.count_nonzero(result[mask] != original[mask]))
    if not changed:
        message = "模型未生成可见的开关变化，原图已保留。" if kind == "toggle" else "模型未生成可见的局部变化，原图已保留。"
        raise StableEditError(message)
    before = hashlib.sha256(original[~mask].tobytes()).hexdigest()
    after = hashlib.sha256(result[~mask].tobytes()).hexdigest()
    outside_changed = int(np.count_nonzero(result[~mask] != original[~mask]))
    if outside_changed or before != after:
        raise StableEditError("局部编辑未通过外部像素一致性检查，原图已保留。")
    output = BytesIO()
    Image.fromarray(result).save(output, format="PNG", compress_level=2, optimize=False)
    # Check the encoded deliverable too, not just an intermediate array.
    delivered = _decode_png(output.getvalue())
    if not np.array_equal(delivered[~mask], original[~mask]):
        raise StableEditError("输出图片未通过外部像素一致性检查，原图已保留。")
    evidence = {"kind": kind, "mode": mode, "bounds": [x0, y0, x1, y1],
                "changed_pixels": changed, "outside_changed_pixels": outside_changed,
                "outside_pixels": int(np.count_nonzero(~mask)),
                "outside_sha256_before": before, "outside_sha256_after": after,
                "verified": True, "verification": "outside_pixels_exact",
                "state_verified": False}
    return output.getvalue(), evidence


def _components(binary):
    """Eight-connected components of a small native switch crop."""
    height, width = binary.shape
    visited = np.zeros(binary.shape, dtype=bool)
    result = []
    for y, x in zip(*np.nonzero(binary)):
        if visited[y, x]:
            continue
        visited[y, x] = True
        stack, points = [(int(y), int(x))], []
        while stack:
            yy, xx = stack.pop()
            points.append((yy, xx))
            for ny in range(max(0, yy - 1), min(height, yy + 2)):
                for nx in range(max(0, xx - 1), min(width, xx + 2)):
                    if binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        points = np.asarray(points, dtype=np.int32)
        ys, xs = points[:, 0], points[:, 1]
        result.append({"points": points, "x0": int(xs.min()), "y0": int(ys.min()),
            "x1": int(xs.max()) + 1, "y1": int(ys.max()) + 1, "area": len(points),
            "cx": float(xs.mean()), "cy": float(ys.mean())})
    return result


def _component_mask(component):
    width, height = component["x1"] - component["x0"], component["y1"] - component["y0"]
    pixels = np.zeros((height, width), dtype=bool)
    points = component["points"]
    pixels[points[:, 0] - component["y0"], points[:, 1] - component["x0"]] = True
    return pixels


def _round_thumb(component, track):
    """Require a compact disk, rather than a line, text stroke, ring or square."""
    width, height = component["x1"] - component["x0"], component["y1"] - component["y0"]
    track_height = track["y1"] - track["y0"]
    if (min(width, height) < 3 or not 0.65 <= width / height <= 1.5
            or not 0.35 * track_height <= height <= 0.9 * track_height):
        return False
    fill = component["area"] / (width * height)
    if not 0.45 <= fill <= 0.92:
        return False
    pixels = _component_mask(component)
    rows = pixels.sum(axis=1)
    largest = int(rows.max())
    # A circular thumb tapers at its top and bottom. This rejects solid
    # rectangles even if their centroid happens to be at the desired side.
    if rows[0] > 0.85 * largest or rows[-1] > 0.85 * largest:
        return False
    center_y = (component["cy"] - track["y0"]) / track_height
    return 0.2 <= center_y <= 0.8


def measure_toggle_state(png, bounds):
    """Return on/off for a recognizable pill+single round thumb, else None.

    The detector uses the pixels and supplied region only. It does not know a
    screen position, seed, prompt or file name. A left disk means off and a
    right disk means on, including globally inverted monochrome palettes.
    Missing tracks, duplicate thumbs and ambiguous geometry fail closed.
    """
    image = _decode_png(png)
    height, width = image.shape
    x0, y0, x1, y1 = _bounds(bounds, width, height)
    crop = image[y0:y1, x0:x1]
    # Infer the local page background from an exterior ring. It only informs
    # classification and is never written back. At image edges use crop corners.
    ex0, ey0, ex1, ey1 = max(0, x0 - 2), max(0, y0 - 2), min(width, x1 + 2), min(height, y1 + 2)
    ring = image[ey0:ey1, ex0:ex1]
    ring_mask = np.ones(ring.shape, dtype=bool)
    ring_mask[y0-ey0:y1-ey0, x0-ex0:x1-ex0] = False
    background_pixels = ring[ring_mask]
    if not len(background_pixels):
        background_pixels = crop[[0, 0, -1, -1], [0, -1, 0, -1]]
    background = float(np.median(background_pixels))
    if 64 < background < 192 or float(np.quantile(crop, 0.9) - np.quantile(crop, 0.1)) < 120:
        return None
    foreground = crop < 128 if background >= 192 else crop >= 128
    components = _components(foreground)
    tracks = []
    crop_height, crop_width = crop.shape
    for component in components:
        cw, ch = component["x1"] - component["x0"], component["y1"] - component["y0"]
        if (cw < 0.78 * crop_width or ch < 0.65 * crop_height
                or not 1.3 <= cw / ch <= 4.5 or component["area"] < 0.06 * cw * ch):
            continue
        pixels = _component_mask(component)
        # Rounded pill ends occupy part of each end column, while the top and
        # bottom must still belong to the same spanning connected component.
        if (pixels[:, 0].sum() > 0.7 * ch or pixels[:, -1].sum() > 0.7 * ch
                or pixels[0].sum() < 0.2 * cw or pixels[-1].sum() < 0.2 * cw):
            continue
        tracks.append(component)
    if len(tracks) != 1:
        return None
    track = tracks[0]
    tw, th = track["x1"] - track["x0"], track["y1"] - track["y0"]
    # Filled foreground track: its thumb is a single enclosed background disk.
    enclosed, enclosed_significant = [], []
    for component in _components(~foreground):
        if (component["x0"] <= track["x0"] or component["x1"] >= track["x1"]
                or component["y0"] <= track["y0"] or component["y1"] >= track["y1"]):
            continue
        if component["area"] >= 3:
            enclosed_significant.append(component)
        if _round_thumb(component, track):
            enclosed.append(component)
    # Empty track: its foreground thumb must be disconnected from its outline.
    separate, separate_significant = [], []
    for component in components:
        if component is track:
            continue
        if (component["x0"] <= track["x0"] or component["x1"] >= track["x1"]
                or component["y0"] <= track["y0"] or component["y1"] >= track["y1"]):
            continue
        if component["area"] >= 3:
            separate_significant.append(component)
        if _round_thumb(component, track):
            separate.append(component)
    if len(enclosed) + len(separate) != 1:
        return None
    thumb = (enclosed or separate)[0]
    position = (thumb["cx"] - track["x0"]) / tw
    fill = track["area"] / (tw * th)
    if (enclosed and len(enclosed_significant) == 1 and not separate_significant
            and fill >= 0.45 and 0.62 <= position <= 0.95):
        return "on"
    if separate and len(separate_significant) == 1 and fill <= 0.5 and 0.05 <= position <= 0.38:
        # The track itself must remain a thin outline. A second foreground
        # thumb can touch that outline and become part of its component; reject
        # the extra solid interior rather than overlooking the duplicate disk.
        points = track["points"].astype(np.float32)
        yy, xx = points[:, 0] - track["y0"], points[:, 1] - track["x0"]
        radius = (th - 1) / 2
        center_x = np.clip(xx, radius, tw - 1 - radius)
        distance = np.sqrt((xx - center_x) ** 2 + (yy - radius) ** 2)
        if np.count_nonzero(distance < radius - 2) > max(3, 0.03 * tw * th):
            return None
        return "off"
    return None


def find_toggle_at(png, click, frame=None):
    """Locate one conservatively measurable switch around an actual click.

    This is read-only pixel geometry, not OCR or a screen/action map. Only a
    160x96 neighborhood is searched, in both monochrome polarities. Every
    candidate must independently pass the existing track/thumb state detector
    and strict edit bounds validation. Invalid inputs or multiple plausible
    targets return None so this optional hint cannot trigger an edit by itself.
    """
    if not isinstance(click, dict) or click.get("type", "click") != "click":
        return None
    x, y = click.get("x"), click.get("y")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or (isinstance(value, float) and not math.isfinite(value))
           for value in (x, y)):
        return None
    try:
        image = _decode_png(png)
        height, width = image.shape
        if max(width, height) > 1024 or not (0 <= x < width and 0 <= y < height):
            return None
        if frame is not None and _dimensions(frame) != (width, height):
            return None
    except StableEditError:
        return None
    dimensions = {"width": width, "height": height}
    event = {"type": "click", "x": x, "y": y}
    cx, cy = math.floor(x), math.floor(y)
    rx0, ry0 = max(0, cx - 80), max(0, cy - 48)
    rx1, ry1 = min(width, cx + 80), min(height, cy + 48)
    crop = image[ry0:ry1, rx0:rx1]
    found = {}
    for binary in (crop < 128, crop >= 128):
        for component in _components(binary):
            cw, ch = component["x1"] - component["x0"], component["y1"] - component["y0"]
            if not (12 <= cw <= 64 and 6 <= ch <= 36 and 1.3 <= cw / ch <= 4.5):
                continue
            # A component cut by the search edge could just be a fragment of
            # a much larger drawing. Image edges themselves remain legal.
            if ((component["x0"] == 0 and rx0 > 0) or
                    (component["y0"] == 0 and ry0 > 0) or
                    (component["x1"] == crop.shape[1] and rx1 < width) or
                    (component["y1"] == crop.shape[0] and ry1 < height)):
                continue
            track = (component["x0"] + rx0, component["y0"] + ry0,
                     component["x1"] + rx0, component["y1"] + ry0)
            if not (track[0] - 1 <= x < track[2] + 1 and track[1] - 1 <= y < track[3] + 1):
                continue
            # Prefer a one-pixel exterior margin; exact component bounds are
            # a fallback for maximal-size tracks or targets at image edges.
            for margin in (1, 0):
                bounds = [max(0, track[0] - margin), max(0, track[1] - margin),
                          min(width, track[2] + margin), min(height, track[3] + margin)]
                try:
                    state = measure_toggle_state(png, bounds)
                    if state is None:
                        continue
                    candidate = validate_toggle_plan({"kind": "toggle", "bounds": bounds,
                        "current_state": state, "target_state": "off" if state == "on" else "on"}, dimensions, event)
                except StableEditError:
                    continue
                candidate["source"] = "pixel_geometry"
                found[track] = candidate
                break
    if not found:
        return None
    # A 2px outline can yield both an outer-track component and its enclosed
    # opposite-polarity interior. These are the same measured target when the
    # states agree and their edges differ by no more than the outline width.
    # Keep the outer boundary, including the whole track, rather than rejecting
    # a legitimate native512 switch as two independent controls.
    candidates = sorted(found.values(), key=lambda candidate:
        (candidate["bounds"][2] - candidate["bounds"][0]) *
        (candidate["bounds"][3] - candidate["bounds"][1]), reverse=True)
    chosen = candidates[0]
    outer = chosen["bounds"]
    for other in candidates[1:]:
        inner = other["bounds"]
        if (other["current_state"] != chosen["current_state"] or
                not (outer[0] <= inner[0] < inner[2] <= outer[2]
                     and outer[1] <= inner[1] < inner[3] <= outer[3]) or
                max(abs(a - b) for a, b in zip(outer, inner)) > 3):
            return None
    return chosen


def validate_toggle_transition(before_png, after_png, edit):
    """Reject unmeasurable, unchanged or contrary states, even if an AI agrees."""
    if not isinstance(edit, dict) or edit.get("kind") != "toggle":
        raise StableEditError("未得到有效的开关编辑信息，原图已保留。")
    current, target = edit.get("current_state"), edit.get("target_state")
    if current not in ("on", "off") or target not in ("on", "off") or current == target:
        raise StableEditError("开关前后状态不明确，原图已保留。")
    if _decode_png(before_png).shape != _decode_png(after_png).shape:
        raise StableEditError("开关验证图片尺寸不一致，原图已保留。")
    before = measure_toggle_state(before_png, edit.get("bounds"))
    after = measure_toggle_state(after_png, edit.get("bounds"))
    if before != current or after != target or before == after:
        raise StableEditError("像素检查未确认单个开关从原状态翻转，原图已保留。")
    return {"pixel_state_verified": True, "before_state": before, "after_state": after,
            "pixel_state_verification": "connected_track_and_single_round_thumb"}
