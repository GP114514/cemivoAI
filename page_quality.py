"""Bounded structural verification before publishing a new generated app page.

This pipeline only inspects complete, actual PNGs. It never edits pixels, commits
application state, executes business tools, or verifies OCR or task completion.
The caller owns the transaction and retains its old page if this function raises.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from time import perf_counter

from initial_review import review_initial
from live_app import LiveError


_SCOPE = "screen_type_empty_state_and_main_control_shapes"
_DRAW_TIMINGS = ("sample_ms", "load_ms", "encode_ms")


def _stage(planner, value):
    callback = getattr(planner, "stage_callback", None)
    if callable(callback):
        callback(value)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _sample_timings(frame):
    timings = frame.get("timings", {})
    if not isinstance(timings, dict):
        return {}
    return {key: timings[key] for key in _DRAW_TIMINGS if _number(timings.get(key))}


def _application_spec(frame, prompt, goal):
    application = frame.get("application")
    if isinstance(application, dict) and isinstance(application.get("spec"), dict):
        return deepcopy(application["spec"])
    # A caller can provide an already planned full specification directly.
    if isinstance(prompt, dict):
        return deepcopy(prompt)
    # Older globally drawn pages have no structured application plan. Preserve
    # their complete supplied instructions instead of inventing a screen class
    # or manufacturing controls from a fixed template.
    return {"initial_prompt": prompt, "original_goal": goal}


def check_page(planner, png, frame, prompt, goal, redraw):
    """Return a structurally accepted candidate, or raise without publishing it.

    ``redraw(caption, seed, steps)`` must return one new actual ``(png, frame)``.
    It is called at most once, only after a valid visual rejection. Network,
    authentication, malformed-review and drawing errors propagate unchanged.
    Each candidate receives one full-image review using the same complete goal
    and specification. ``ui_quality_check is True`` explicitly enables this
    path; legacy fixtures and disabled planners retain their original behavior.
    """
    if getattr(planner, "ui_quality_check", False) is not True:
        return png, frame
    if not isinstance(frame, dict):
        raise LiveError("页面结构检查缺少候选画面，原图已保留。", "initial_review_failed", 502)

    candidate = deepcopy(frame)
    application_spec = _application_spec(candidate, prompt, goal)
    cumulative = _sample_timings(candidate)
    review_ms = 0.0
    review_attempts = 0
    checks = []
    repairs = 0

    def evidence(accepted=False):
        return {"accepted": accepted, "attempts": review_attempts,
            "repair_attempts": repairs, "checks": deepcopy(checks),
            "scope": _SCOPE, "readability": "unknown",
            "business_completion": "not_assessed",
            "review_ms": round(review_ms, 2)}

    def timings():
        result = deepcopy(candidate.get("timings", {}))
        if not isinstance(result, dict):
            result = {}
        result.update({key: round(value, 2) for key, value in cumulative.items()})
        result["page_review_ms"] = round(review_ms, 2)
        return result

    try:
        for attempt in range(2):
            _stage(planner, "verifying")
            review_attempts += 1
            started = perf_counter()
            try:
                checked = review_initial(planner, png, candidate, application_spec, goal)
            finally:
                review_ms += (perf_counter() - started) * 1000
            check = deepcopy(checked)
            check["image_sha256"] = hashlib.sha256(png).hexdigest()
            checks.append(check)
            if checked["matches_screen"] and checked["main_controls_present"]:
                candidate["page_review"] = evidence(accepted=True)
                candidate["timings"] = timings()
                return png, candidate
            if attempt == 1:
                reason = " ".join(checked["reason"].split())[:120]
                raise LiveError("页面经一次重画仍不符合要求：" + reason + "；原图已保留。",
                    "page_mismatch", 422)

            seed, steps = candidate.get("seed"), candidate.get("steps")
            if (type(seed) is not int or not 0 <= seed < 2**31
                    or type(steps) is not int or steps not in (4, 8, 16, 32)):
                raise LiveError("页面修复缺少有效绘图参数，原图已保留。", "initial_review_failed", 502)
            _stage(planner, "drawing")
            repairs += 1
            repaired_png, repaired_frame = redraw(checked["repair_prompt"], (seed + 1) % 2**31, steps)
            if not isinstance(repaired_frame, dict):
                raise LiveError("修复画面的回执不完整，原图已保留。", "initial_review_failed", 502)
            for key, value in _sample_timings(repaired_frame).items():
                cumulative[key] = cumulative.get(key, 0) + value
            # Preserve the caller's candidate context and draft, while accepting
            # only the redraw callback's actual image identity/model metadata.
            candidate.update(deepcopy(repaired_frame))
            png = repaired_png
    except LiveError as error:
        # Preserve the original error/code (especially network/auth). The caller
        # may journal inspection evidence but must not commit a rejected image.
        if isinstance(getattr(error, "review_evidence", None), dict):
            check = deepcopy(error.review_evidence)
            check["image_sha256"] = hashlib.sha256(png).hexdigest()
            checks.append(check)
        error.page_review = evidence()
        error.page_review_timings = timings()
        raise
