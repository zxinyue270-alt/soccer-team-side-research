"""Optional visual explanation adapter; never stores credentials or invents API results."""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gbf.util.hashing import sha256_file
from gbf.util.io import write_json

TITLES = {
    "duel_tackle": "对抗回放",
    "sprint": "跑动回放",
    "ball_strike": "球路变化回放",
    "attack_in_box": "禁区附近回放",
}


def guidance(kind: str, *, local_available: bool) -> list[str]:
    """Replay guidance, deliberately not a claim that a model understood an event."""
    opening = {
        "duel_tackle": "先看原速回放，留意球员接近时的位置变化。",
        "sprint": "先看原速回放，留意球员之间的位置变化。",
        "ball_strike": "先看原速回放，留意球员动作与足球的去向。",
        "attack_in_box": "先看原速回放，留意进攻区域的跑位与防守位置。",
    }.get(kind, "先看这段原速回放，观察球员和足球的位置变化。")
    replay = (
        "接下来慢放回看。结合全景与局部，观察身体姿态和周围空间。"
        if local_available
        else "接下来慢放回看。留意全景中的脚下动作和周围球员的跑位。"
    )
    return [opening, replay]


def validate_answer(answer: Any, supplied: set[int]) -> list[str]:
    if not isinstance(answer, dict) or set(answer) != {"lines"}:
        raise ValueError("Explanation must contain lines only")
    lines = answer["lines"]
    if not isinstance(lines, list) or len(lines) != 2:
        raise ValueError("Exactly two narration lines are required")
    adopted = []
    for line in lines:
        if not isinstance(line, dict) or set(line) != {"text", "frames", "kind"}:
            raise ValueError("Malformed explanation line")
        text, frames = line["text"], line["frames"]
        if (
            not isinstance(text, str)
            or not 5 <= len(text) <= 44
            or any(ord(c) < 32 for c in text)
            or not isinstance(frames, list)
            or not frames
            or any(type(f) is not int or f not in supplied for f in frames)
            or line["kind"] not in {"observation", "hypothesis"}
        ):
            raise ValueError("Invalid text length, evidence frames or claim kind")
        if re.search(r"\d\s*(米|公里|km|m/s|度)|sk-[A-Za-z0-9_-]{12,}", text):
            raise ValueError("Unverified measurement or credential-like text")
        if line["kind"] == "hypothesis" and not any(
            word in text for word in ("可能", "似乎", "或许", "推测")
        ):
            raise ValueError("Tactical hypothesis must be qualified")
        adopted.append(text)
    return adopted


def explain(
    samples: Sequence[dict[str, Any]],
    kind: str,
    output: Path,
    *,
    local_available: bool = False,
) -> dict[str, Any]:
    """At most one request per selected clip; errors fall back with an explicit source."""
    fallback = {
        "source": "local_replay_guidance",
        "lines": guidance(kind, local_available=local_available),
        "tactical_explanation_generated": False,
        "accuracy_certified": False,
        "reason": "provider_not_configured",
    }
    key = os.environ.get("GBF_DEEPSEEK_API_KEY", "")
    if not key:
        write_json(output / "explanation.json", fallback)
        return fallback
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "以下全景按时间顺序排列，每张后可能附同帧局部。局部只裁剪，不是完整画面。"
            "先识别这段真正可见的重要变化，例如争顶、倒地、起身、跑动、门将反应；"
            "不要忽略持续可见的倒地球员。不得将检测缺失等同画面不可见。"
            "每个frame为源帧号，t为本段相对秒，boxes为局部观察对象的估计框，"
            "框缺失就不连接身份。原始场景："
            + json.dumps([s["evidence"] for s in samples], ensure_ascii=False),
        }
    ]
    for sample in samples:
        for name in ("full", "detail"):
            if sample.get(name) is None:
                continue
            content.extend(
                [
                    {"type": "text", "text": f"源帧{sample['frame']} {name}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(sample[name]).decode("ascii"),
                            "detail": "original",
                        },
                    },
                ]
            )
    payload = {
        "model": "deepseek-flash",
        "messages": [
            {
                "role": "system",
                "content": "为足球回放写两句中文旁白。第一句配原速，第二句以‘慢放回看’开始，"
                "描述最明显的动作或有依据的空间关系。每句5到44字，总计不超过88字。"
                "仅依据给出的抽样图，不声称看过中间帧。不要套通用边路推进模板。"
                "球权、触球、射门结果、接触原因或犯规不能确认就省略。"
                "不要实名、号码猜测、精确测量、医疗判断或确定战术意图。"
                "战术推测必须写‘可能’，仍须指明画面依据。提供两句即可："
                '{"lines":[{"text":"第一句","frames":[源帧号],"kind":"observation"},'
                '{"text":"慢放回看……","frames":[源帧号],"kind":"hypothesis"}]}。'
                "允许两句都是observation。输出JSON，不要其他字段。",
            },
            {"role": "user", "content": content},
        ],
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "max_tokens": 700,
        "stream": False,
    }
    write_json(output / "model-request.json", payload)
    payload_bytes = (output / "model-request.json").read_bytes()
    json.loads(payload_bytes.decode("utf-8"))
    if b"\xef\xbf\xbd" in payload_bytes or b"???" in payload_bytes or key.encode() in payload_bytes:
        raise ValueError("Invalid or sensitive request content")
    write_json(
        output / "model-preflight.json",
        {
            "passed": True,
            "payload_sha256": sha256_file(output / "model-request.json"),
            "credentials_in_payload": False,
            "source_frames": [s["frame"] for s in samples],
            "provenance": "Canonical source frames; detail is same-frame crop from estimated boxes",
            "requirements_trace": {
                "preserve_context": "full-scene images in chronological order",
                "extract_action_details": ("same-frame detail images when tracking is available"),
                "avoid_certain_tactical_intent": "qualified hypotheses; source-frame references",
            },
            "requests_in_this_part": 1,
            "max_completion_tokens": 700,
        },
    )
    try:
        request = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=payload_bytes,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read(1_000_000).decode("utf-8")
        raw = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[REDACTED]", raw.replace(key, "[REDACTED]"))
        data = json.loads(raw)
        write_json(output / "model-response.json", data)
        if data["choices"][0]["finish_reason"] != "stop":
            raise ValueError("Incomplete model response")
        answer = json.loads(data["choices"][0]["message"]["content"])
        lines = validate_answer(answer, {s["frame"] for s in samples})
        result = {
            "source": "deepseek-flash",
            "lines": lines,
            "claims": answer["lines"],
            "usage": data.get("usage"),
            "tactical_explanation_generated": True,
            "semantic_review": "not_independently_verified",
            "accuracy_certified": False,
        }
        write_json(output / "explanation.json", result)
        return result
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        fallback["reason"] = "provider_or_contract_error:" + type(exc).__name__
        write_json(output / "explanation.json", fallback)
        return fallback
