"""Read-only pixel evidence and a precisely mapped click crop for vision.

The original screenshot is always retained as the first image. Enlarging the
crop improves attention to existing pixels, never invents detail or renders UI.
"""
from io import BytesIO
import math

from PIL import Image, UnidentifiedImageError


def prepare_click_view(png, frame, event):
    if not isinstance(event, dict) or event.get("type") != "click":
        return {}, []
    if not isinstance(frame, dict) or any(type(frame.get(key)) is not int or not 32 <= frame[key] <= 1024 for key in ("width", "height")):
        raise ValueError("Invalid native frame dimensions")
    x, y = event.get("x"), event.get("y")
    width, height = frame["width"], frame["height"]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (x, y)) or not (0 <= x < width and 0 <= y < height):
        raise ValueError("Click must be inside the original image")
    if not isinstance(png, bytes) or not 0 < len(png) <= 8 * 1024**2:
        raise ValueError("Invalid screenshot bytes")
    try:
        with Image.open(BytesIO(png)) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (width, height) or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Screenshot does not match native frame")
            image.load()
            from stable_edit import find_toggle_at
            control = find_toggle_at(png, {"x":x,"y":y}, frame)
            from control_grounding import find_button_at
            button = None if control else find_button_at(png,{'x':x,'y':y},frame)
            half = min(64, max(24, min(width, height) // 5))
            x0, y0 = max(0, int(x)-half), max(0, int(y)-half)
            x1, y1 = min(width, int(x)+half+1), min(height, int(y)+half+1)
            # Shift crops at the edges rather than padding fabricated pixels.
            x0, y0 = max(0, x1-(2*half+1)), max(0, y1-(2*half+1))
            x1, y1 = min(width, x0+2*half+1), min(height, y0+2*half+1)
            if button:
                bx0,by0,bx1,by1=button['bounds']
                x0,y0,x1,y1=max(0,bx0-5),max(0,by0-5),min(width,bx1+5),min(height,by1+5)
            scale = 3
            crop = image.crop((x0, y0, x1, y1))
            low, high = crop.getextrema()
            focus = crop.resize(((x1-x0)*scale, (y1-y0)*scale), Image.Resampling.NEAREST)
            output = BytesIO()
            focus.save(output, format="PNG", compress_level=2, optimize=False)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise ValueError("Screenshot cannot be decoded") from None
    return {"focus": {"image_index": 2, "bounds": [x0, y0, x1, y1], "scale": scale,
        "width": focus.width, "height": focus.height, "has_visible_detail": high-low >= 60,
        "click_in_focus": [round((x-x0)*scale, 4), round((y-y0)*scale, 4)],
        "mapping": "original_x=x0+focus_x/scale; original_y=y0+focus_y/scale",
        "note": "Nearest-neighbor view of actual source pixels only; bounds refer to ORIGINAL image"},
        "pixel_control": control, 'pixel_button':button}, [output.getvalue()]


VISION_EVIDENCE = """The FIRST image is the exact full screenshot. If present,
the SECOND image is a nearest-neighbor enlarged crop of the clicked area.
focus metadata supplies the exact crop bounds, scale and clicked position.
pixel_button, when present, is the smallest measured closed button/input box
containing the click. Its label is UNKNOWN: read the real pixels. The focus is
tight to that enclosure; do not substitute a neighboring control above/below.
For an unboxed text item there may be no pixel_button: use the exact click
coordinates and full screenshot rather than assuming it is a nearby button.
Return ALL control bounds in ORIGINAL screenshot coordinates, never crop or
enlarged coordinates. Zoom does not create detail. Read both views together.
pixel_control, when present, is an independently measured recognizable rounded
track and single circular thumb at the click; its geometry/state comes from
actual pixels, not labels or the intended prompt. A visible toggle can be used
without knowing its garbled text label: describe it by location. For ordinary
toggle clicks use that measured state/bounds instead of guessing from color.
Do not invent unreadable text, hidden targets or a successful business action.
When no action can be identified, provide a concise concrete clarification in
SIMPLIFIED CHINESE: describe what was visible, what is unclear, and what brief
instruction the user can enter. Do not return only 'uncertain' or 'no change'.
All user-facing intent, expected_change, state_summary and scope_reason should
use Simplified Chinese; the compact next_prompt may remain English.
"""
