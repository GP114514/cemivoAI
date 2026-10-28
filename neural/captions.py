"""Metadata-grounded natural captions for training, never image generation.

The caller decides whether to keep the original caption (for example 40%) or
call augment_caption (60%). Only the supplied NumPy Generator is advanced.
The change is a testable generalization hypothesis, not a measured improvement.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import re

import numpy as np


MAX_UTF8_BYTES = 190
VERSION = "metadata-grounded-natural-captions-v1"

# Aliases name the same visible family; they do not promise extra controls.
ALIASES = {
    "chat": {
        "zh": ("聊天窗口", "聊天界面", "聊天页面", "聊天软件界面", "对话界面", "消息窗口"),
        "en": ("chat window", "chat app", "chat screen", "messaging interface", "conversation window", "chat interface"),
    },
    "login": {
        "zh": ("登录页面", "登录界面", "登录窗口", "账号登录页", "登录表单", "登录页"),
        "en": ("login page", "sign in page", "sign-in screen", "login form", "account sign-in screen", "sign in screen"),
    },
    "dashboard": {
        "zh": ("数据面板", "数据仪表盘", "统计看板", "数据概览界面", "统计面板", "后台数据页"),
        "en": ("dashboard", "data dashboard", "analytics panel", "statistics screen", "data overview", "statistics dashboard"),
    },
    "settings": {
        "zh": ("设置页面", "设置界面", "设置面板", "偏好设置页", "应用设置页", "系统设置界面"),
        "en": ("settings panel", "settings screen", "settings page", "preferences window", "preferences panel", "app settings screen"),
    },
    "files": {
        "zh": ("文件管理器", "文件管理界面", "文件浏览窗口", "文件列表页面", "文件浏览器", "文件管理窗口"),
        "en": ("file manager", "file browser", "file management screen", "files page", "folder browser", "file manager window"),
    },
    "shop": {
        "zh": ("购物页面", "商品页面", "商城界面", "商品列表页", "购物商城", "网店界面"),
        "en": ("shopping page", "online store", "product catalog", "shop interface", "product listing", "store page"),
    },
    "music": {
        "zh": ("音乐播放器", "音乐界面", "歌曲播放页", "音乐播放窗口", "听歌界面", "音乐软件界面"),
        "en": ("music player", "music app", "song player", "audio player", "music screen", "music player window"),
    },
    "calendar": {
        "zh": ("日历页面", "日历界面", "月历页面", "月历窗口", "日程日历", "日历面板"),
        "en": ("calendar", "calendar page", "monthly calendar", "calendar screen", "month view", "calendar window"),
    },
    "editor": {
        "zh": ("文档编辑器", "编辑器界面", "文本编辑器", "文档编辑页面", "写作界面", "编辑窗口"),
        "en": ("document editor", "text editor", "writing interface", "document editing screen", "editor window", "writing app"),
    },
    "mail": {
        "zh": ("电子邮箱", "邮箱界面", "收件箱页面", "邮件列表页", "邮件客户端", "邮件界面"),
        "en": ("email inbox", "mail interface", "email client", "inbox page", "mail screen", "email interface"),
    },
    "browser": {
        "zh": ("网页浏览器", "浏览器界面", "浏览器窗口", "网页浏览界面", "浏览器页面", "网页窗口"),
        "en": ("web browser", "browser window", "browser interface", "web browser screen", "browser page", "web browsing window"),
    },
    "terminal": {
        "zh": ("终端窗口", "命令行界面", "控制台窗口", "命令终端", "终端界面", "命令行窗口"),
        "en": ("terminal window", "command line", "console window", "shell terminal", "terminal interface", "command prompt screen"),
    },
    "diagram": {
        "zh": ("流程图", "流程示意图", "步骤流程图", "流程节点图", "工作流程图", "方框流程图"),
        "en": ("flowchart", "workflow diagram", "process diagram", "workflow chart", "process flowchart", "box-and-arrow diagram"),
    },
    "shapes": {
        "zh": ("基础几何图形", "几何图形", "基础图形", "简单几何线稿", "几何线条图", "基础图形图片"),
        "en": ("basic geometric shapes", "geometric shapes", "basic shapes", "simple geometry drawing", "geometric figure drawing", "simple shapes image"),
    },
    "text": {
        "zh": ("文字海报", "文字排版图", "文字卡片", "标题排版图", "文字展示图片", "排版海报"),
        "en": ("text poster", "typography poster", "text card", "headline layout", "typographic image", "text layout image"),
    },
}

# Each phrase describes a feature drawn in every example of its family in
# dataset.py. Counts, chart types, specific shapes, notification labels,
# contacts, voice calls, tabs, and other unrecorded random choices are omitted.
DETAILS = {
    "chat": {
        "zh": ("消息输入框", "聊天气泡和消息区", "底部操作按钮", "消息输入区域"),
        "en": ("a message input box", "message bubbles and a conversation area", "bottom action buttons", "a message input area"),
    },
    "login": {
        "zh": ("邮箱和密码输入框", "登录表单与输入框", "表单操作按钮", "账号输入区域"),
        "en": ("email and password input fields", "a sign in form with input boxes", "form action buttons", "an account input area"),
    },
    "dashboard": {
        "zh": ("统计卡片和图表", "数据概览与指标卡片", "图表区域", "数据统计区域"),
        "en": ("statistics cards and a chart", "data overview cards", "a chart area", "summary metrics"),
    },
    "settings": {
        "zh": ("开关和选项", "偏好设置内容", "开关选项与底部按钮", "设置选项"),
        "en": ("toggle switches and options", "preference options", "switches and bottom buttons", "settings options"),
    },
    "files": {
        "zh": ("文件搜索输入框", "文件夹卡片和文件列表", "文件名与大小", "文件搜索区域"),
        "en": ("a file search input box", "folder cards and a file list", "file names and sizes", "a file search area"),
    },
    "shop": {
        "zh": ("商品卡片与价格", "商品列表和操作按钮", "商品展示区域", "带价格的商品卡片"),
        "en": ("product cards and prices", "products and action buttons", "a product display area", "product cards with prices"),
    },
    "music": {
        "zh": ("专辑封面和歌曲列表", "播放按钮与进度线", "歌曲信息和播放区域"),
        "en": ("an album cover and a song list", "a play button and a progress line", "song information and playback controls"),
    },
    "calendar": {
        "zh": ("日期网格与月份标题", "星期和日期", "月历网格", "顶部操作按钮"),
        "en": ("a date grid and a month heading", "weekdays and dates", "a monthly grid", "a top action button"),
    },
    "editor": {
        "zh": ("工具栏和文档编辑区", "文档编辑输入框", "编辑输入区域和格式工具", "文档页面与正文线条"),
        "en": ("a toolbar and a document editing area", "a document text input area", "an editing input box and formatting tools", "a document page with body text lines"),
    },
    "mail": {
        "zh": ("收件箱和邮件列表", "发件人和邮件摘要", "邮件条目", "顶部操作按钮"),
        "en": ("an inbox and a mail list", "senders and message summaries", "email entries", "a top action button"),
    },
    "browser": {
        "zh": ("网址输入框和导航符号", "地址栏和网页内容", "网页标题与操作按钮", "底部内容卡片"),
        "en": ("a URL input box and navigation symbols", "an address bar and page content", "a page heading and an action button", "bottom content cards"),
    },
    "terminal": {
        "zh": ("命令行文本和输出", "命令提示符与输入位置", "终端输出内容", "命令文本区域"),
        "en": ("command line text and output", "a command prompt and an input position", "terminal output", "a command text area"),
    },
    "diagram": {
        "zh": ("方框和箭头", "流程节点与连线", "带文字的步骤节点", "节点之间的箭头"),
        "en": ("boxes and arrows", "flow nodes and connections", "labeled process nodes", "arrows between nodes"),
    },
    "shapes": {
        "zh": ("简单几何元素", "图形与文字标签", "基础几何元素与标签"),
        "en": ("simple geometric elements", "geometric figures with text labels", "basic geometry and labels"),
    },
    "text": {
        "zh": ("大字标题和副标题", "居中文字与装饰细线", "文字主标题", "标题文字和留白"),
        "en": ("a large headline and a subtitle", "centered text and a decorative line", "a text headline", "headline text and whitespace"),
    },
}

PREFIXES = {
    "zh": ("给我一个{alias}的UI", "生成一个{alias}", "画一个{alias}", "制作一个{alias}", "我想要{alias}", "设计一个{alias}", "帮我做一个{alias}", "{alias}"),
    "en": ("Give me a {alias} UI", "Create a {alias}", "Draw a {alias}", "Design a {alias}", "I want a {alias}", "Show a {alias}", "Make a {alias}", "{alias}"),
}

STYLES = {
    ("zh", False): ("白底黑线", "白色背景、黑色线条", "黑白线框，白底", "白底的黑白简图"),
    ("zh", True): ("黑底白线", "黑色背景、白色线条", "黑白线框，黑底", "黑底的黑白简图"),
    ("en", False): ("white background with black lines", "black lines on white background", "monochrome on a white background", "a white background and black outlines"),
    ("en", True): ("black background with white lines", "white lines on black background", "monochrome on a black background", "a black background and white outlines"),
}


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _quote(value, language):
    value = str(value)
    if language == "zh" or '"' in value:
        if "“" in value or "”" in value:
            raise ValueError("A caption label containing nested curly quotes cannot be represented safely")
        return "“" + value + "”"
    return '"' + value + '"'


def augment_caption(meta, rng) -> str:
    """Return a <=190-byte truthful variant, advancing only the supplied RNG.

    metadata.buttons means exact visible button labels except for kind=text,
    where it contains the poster headline/subtitle. Optional labels may be
    omitted as complete fields, never shortened into false exact text.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    # The new app targets carry explicit family/empty-body supervision. Route
    # their metadata before the historical aliases or DETAILS can discard it.
    # Keep the historical branch and its RNG draws unchanged for V2/V3 rows.
    if meta.get('dataset_version') == 'native256-app-flows-v5':
        from .state_sampling import app_flow_caption_variants
        variants = app_flow_caption_variants(meta)
        return variants[0 if rng.random() < .5 else 1]
    if meta.get('dataset_version') == 'native256-app-curriculum-v4':
        from .app_curriculum import augment_app_caption
        return augment_app_caption(meta, rng)
    kind = str(meta["kind"])
    if kind not in ALIASES:
        raise ValueError(f"Unsupported image family: {kind}")
    language = "zh" if float(rng.random()) < .65 else "en"
    alias = _choice(rng, ALIASES[kind][language])
    base = _choice(rng, PREFIXES[language]).format(alias=alias)
    style = _choice(rng, STYLES[(language, bool(meta.get("dark", False)))])
    # Priority 100 items are mandatory. Low priority descriptive fields are
    # removed first if byte budgeting requires it; quoted strings stay whole.
    fields = [(100, "family", base), (100, "style", style)]
    if float(rng.random()) < .68:
        count = 1 + int(float(rng.random()) < .22)
        choices = DETAILS[kind][language]
        positions = rng.choice(len(choices), size=min(count, len(choices)), replace=False)
        for position in positions:
            detail = choices[int(position)]
            fields.append((10, "detail", ("有" if language == "zh" else "with ") + detail))
    if bool(meta.get("sidebar", False)) and float(rng.random()) < .55:
        fields.append((20, "sidebar", "左侧有导航栏" if language == "zh" else "a left navigation sidebar"))
    title = str(meta.get("title", ""))
    if title and float(rng.random()) < .57:
        fields.append((40, "title", ("标题" if language == "zh" else "title ") + _quote(title, language)))
    labels = [str(value) for value in meta.get("buttons", []) if str(value)]
    if labels and float(rng.random()) < .65:
        # A subset is explicitly introduced as "includes" and never claims to
        # enumerate all controls or to give their positions/order.
        if len(labels) > 2 and float(rng.random()) < .45:
            amount = int(rng.integers(1, min(3, len(labels)) + 1))
            positions = sorted(int(i) for i in rng.choice(len(labels), size=amount, replace=False))
            labels = [labels[i] for i in positions]
        quoted = ("、" if language == "zh" else ", ").join(_quote(value, language) for value in labels)
        if language == "zh":
            label_intro = "文字包括" if kind == "text" else "节点文字包括" if kind == "diagram" else "按钮文字包括"
        else:
            label_intro = "text includes " if kind == "text" else "node labels include " if kind == "diagram" else "button labels include "
        fields.append((60, "labels", label_intro + quoted))
    # Vary optional field order without making the leading request ungrammatical.
    head, tail = fields[0], fields[1:]
    rng.shuffle(tail)
    fields = [head] + tail
    separator = "，" if language == "zh" else ", "
    while True:
        result = separator.join(field[2] for field in fields)
        if len(result.encode("utf-8")) <= MAX_UTF8_BYTES:
            return result
        removable = [(field[0], index) for index, field in enumerate(fields) if field[0] < 100]
        if not removable:
            raise ValueError("Mandatory family and style fields exceeded the caption byte budget")
        _, index = min(removable)
        fields.pop(index)


def _numpy_state_equal(first, second):
    return first[0] == second[0] and np.array_equal(first[1], second[1]) and first[2:] == second[2:]


def verify(data, tests_per_kind=1000, report_path=None, examples_path=None):
    """CPU-only metadata contract checks; no Torch import or image/model access."""
    data = Path(data)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    by_kind = {kind: [] for kind in ALIASES}
    for shard in manifest["shards"]:
        for line in (data / shard["metadata"]).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["split"] == "train":
                by_kind[row["kind"]].append(row)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    rng = np.random.default_rng(20261025)
    reports = {}
    examples = []
    maximum = 0
    cases = 0
    for kind, rows in by_kind.items():
        if not rows:
            raise ValueError(f"No metadata examples available for {kind}")
        counts = {"tests": 0, "quoted_title_retained": 0, "quoted_labels_retained": 0, "input_area_wording": 0, "unique_captions": 0, "max_utf8_bytes": 0}
        observed = {"zh": set(), "en": set()}
        distinct = set()
        for index in range(tests_per_kind):
            meta = rows[(index * 15485863) % len(rows)]
            caption = augment_caption(meta, rng)
            encoded = caption.encode("utf-8")
            assert len(encoded) <= MAX_UTF8_BYTES
            maximum = max(maximum, len(encoded))
            counts["max_utf8_bytes"] = max(counts["max_utf8_bytes"], len(encoded))
            quotes = [a or b for a, b in re.findall(r'“([^”]*)”|"([^"]*)"', caption)]
            assert len(quotes) == caption.count("“") + caption.count('"') // 2
            allowed = {str(meta["title"])} | {str(value) for value in meta.get("buttons", [])}
            assert set(quotes) <= allowed, (kind, caption, allowed)
            assert all(quotes), caption
            if bool(meta["dark"]):
                assert not any(term in caption for term in ("白底", "白色背景", "white background", "a white background")), caption
                assert any(term in caption for term in ("黑底", "黑色背景", "black background")), caption
            else:
                assert not any(term in caption for term in ("黑底", "黑色背景", "black background", "a black background")), caption
                assert any(term in caption for term in ("白底", "白色背景", "white background")), caption
            if not bool(meta["sidebar"]):
                assert "导航栏" not in caption and "sidebar" not in caption, caption
            assert not any(term in caption.lower() for term in ("联系人列表", "语音按钮", "contacts list", "voice call", "volume slider", "notification toggle"))
            if kind == "text":
                assert "按钮" not in caption and "button labels" not in caption
            for language, aliases in ALIASES[kind].items():
                observed[language].update(alias for alias in aliases if alias in caption)
            counts["quoted_title_retained"] += str(meta["title"]) in quotes
            counts["quoted_labels_retained"] += bool(set(quotes) & set(meta.get("buttons", [])))
            counts["input_area_wording"] += "输入框" in caption or "input box" in caption or "input fields" in caption or "input area" in caption
            counts["tests"] += 1
            cases += 1
            distinct.add(caption)
            if index < 10:
                examples.append({"kind": kind, "source_index": meta["index"], "dark": meta["dark"], "sidebar": meta["sidebar"], "source_title": meta["title"], "source_labels": meta.get("buttons", []), "caption": caption, "utf8_bytes": len(encoded)})
        counts["unique_captions"] = len(distinct)
        counts["aliases_observed"] = {language: sorted(values) for language, values in observed.items()}
        assert all(set(ALIASES[kind][language]) <= observed[language] for language in ("zh", "en")), kind
        if kind in ("chat", "login", "files", "editor", "browser"):
            assert counts["input_area_wording"] > 0, kind
        reports[kind] = counts
    # Check byte-identical continuation with the caller-owned saved RNG state.
    recovery_rng = np.random.default_rng(23)
    sequence = [by_kind[kind][0] for kind in ALIASES] * 3
    state = copy.deepcopy(recovery_rng.bit_generator.state)
    first = [augment_caption(meta, recovery_rng) for meta in sequence]
    after_first = copy.deepcopy(recovery_rng.bit_generator.state)
    recovery_rng.bit_generator.state = state
    second = [augment_caption(meta, recovery_rng) for meta in sequence]
    assert first == second
    assert json.dumps(after_first, sort_keys=True) == json.dumps(recovery_rng.bit_generator.state, sort_keys=True)
    # Oversized exact strings are omitted as whole fields, never shortened.
    huge = dict(by_kind["chat"][0], title="长标题" * 150, buttons=["很长的精确按钮文字" * 80])
    for _ in range(100):
        caption = augment_caption(huge, recovery_rng)
        assert len(caption.encode("utf-8")) <= MAX_UTF8_BYTES
        assert not re.findall(r'“([^”]*)”|"([^"]*)"', caption)
    assert random.getstate() == python_before
    assert _numpy_state_equal(np.random.get_state(), numpy_before)
    report = {"version": VERSION, "passed": True, "test_cases": cases, "tests_per_kind": tests_per_kind, "families": len(reports), "max_output_utf8_bytes": maximum, "byte_limit": MAX_UTF8_BYTES, "global_python_rng_unchanged": True, "global_numpy_rng_unchanged": True, "saved_rng_resume_byte_identical": True, "quoted_labels_are_metadata_subsets": True, "dark_light_consistent": True, "oversized_exact_fields_dropped_whole": True, "source_dataset": str(data.resolve()), "per_kind": reports, "limitations": ["These checks establish metadata/caption contracts, not image generation quality or improved generalization.", "Generic feature phrases were reviewed against dataset.py drawing branches; no new images or OCR were used.", "Unrecorded choices such as exact setting option names, contact lists, call controls, chart type or shape type are never promised.", "The caller controls the mixture with original captions; recommended 60% augmentation is not applied by this module.", "Actual model improvements require checkpoint-controlled training and inference comparisons."]}
    if report_path:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if examples_path:
        examples_path = Path(examples_path)
        examples_path.parent.mkdir(parents=True, exist_ok=True)
        examples_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="neural/data-v2")
    parser.add_argument("--tests-per-kind", type=int, default=1000)
    parser.add_argument("--report", default="work/augmentation-report.json")
    parser.add_argument("--examples", default="work/augmentation-examples.json")
    args = parser.parse_args()
    if args.tests_per_kind < 1000:
        parser.error("Contract verification requires at least 1000 examples per family")
    report = verify(args.data, args.tests_per_kind, args.report, args.examples)
    print(json.dumps({key: report[key] for key in ("version", "passed", "test_cases", "families", "max_output_utf8_bytes", "global_python_rng_unchanged", "global_numpy_rng_unchanged", "saved_rng_resume_byte_identical")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
