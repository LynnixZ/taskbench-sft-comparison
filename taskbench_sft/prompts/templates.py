"""Static prompt templates and the shared one-shot example.

The system prompt, tool catalog, one-shot example, and user request are kept
identical across the two modes; only the *output task description* differs. The
one-shot example is the same fixed example for both modes (only its answer format
changes), so neither mode receives extra demonstrations.
"""
from __future__ import annotations

from taskbench_sft.targets import canonical_json

# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #
SYSTEM_FULL_JSON = (
    "You are a tool-planning model. Given a user request and a catalog of "
    "available tools, produce a valid plan using only tools from the catalog. "
    "Do not invent tool names. Return only the required JSON object, without "
    "markdown or explanation."
)

SYSTEM_TRAJECTORY = (
    "You are a tool-planning model. Given a user request and a catalog of "
    "available tools, produce the ordered tool trajectory required to complete "
    "the request. Use only tools from the catalog. Return only the required JSON "
    "array, without markdown or explanation."
)

# --------------------------------------------------------------------------- #
# Shared one-shot example (same request + catalog for both modes)
# --------------------------------------------------------------------------- #
_ONE_SHOT_TOOLS = [
    {"id": "Image-to-Text", "description": "Extract text from image"},
    {"id": "Translation", "description": "Translate text"},
    {"id": "Text-to-Speech", "description": "Convert text to speech"},
]
_ONE_SHOT_REQUEST = "Translate the text in an image into French and read it aloud."

_ONE_SHOT_FULL_JSON_ANSWER = canonical_json(
    {
        "task_steps": [
            "Extract text from the image",
            "Translate the extracted text into French",
            "Convert the translated text to speech",
        ],
        "task_nodes": [
            {"task": "Image-to-Text", "arguments": []},
            {"task": "Translation", "arguments": []},
            {"task": "Text-to-Speech", "arguments": []},
        ],
        "task_links": [
            {"source": "Image-to-Text", "target": "Translation"},
            {"source": "Translation", "target": "Text-to-Speech"},
        ],
    }
)

_ONE_SHOT_TRAJECTORY_ANSWER = canonical_json(
    ["Image-to-Text", "Translation", "Text-to-Speech"]
)


def _one_shot_catalog_str() -> str:
    import json

    lines = [json.dumps(e, ensure_ascii=False) for e in _ONE_SHOT_TOOLS]
    return "[\n" + ",\n".join("  " + ln for ln in lines) + "\n]"


def one_shot_block(answer: str) -> str:
    """Render the one-shot example: a request, its catalog, and the answer."""
    return (
        "AVAILABLE TOOLS:\n"
        f"{_one_shot_catalog_str()}\n\n"
        f"USER REQUEST:\n{_ONE_SHOT_REQUEST}\n\n"
        f"ANSWER:\n{answer}"
    )


ONE_SHOT_FULL_JSON = one_shot_block(_ONE_SHOT_FULL_JSON_ANSWER)
ONE_SHOT_TRAJECTORY = one_shot_block(_ONE_SHOT_TRAJECTORY_ANSWER)

# --------------------------------------------------------------------------- #
# User templates
# --------------------------------------------------------------------------- #
USER_FULL_JSON = """AVAILABLE TOOLS:
{catalog}
{example_block}USER REQUEST:
{user_request}

Generate a complete task automation plan.

Return exactly one JSON object with these top-level keys:
- "task_steps"
- "task_nodes"
- "task_links"

Requirements:
1. Each task step must align with exactly one task node.
2. Every selected tool must come from AVAILABLE TOOLS.
3. task_links must represent the execution dependencies.
4. For a single-tool task, task_links must be an empty list.
5. Include the required tool arguments using the TaskBench schema.
6. Return JSON only."""

USER_TRAJECTORY = """AVAILABLE TOOLS:
{catalog}
{example_block}USER REQUEST:
{user_request}

Return the ordered tool trajectory needed to complete the request.

Requirements:
1. Return one JSON array of exact tool IDs.
2. The order must be the execution order.
3. Use only tools from AVAILABLE TOOLS.
4. Do not include explanations, task steps, arguments, or extra keys.
5. For a single-tool request, return an array containing one tool.
6. Return JSON only."""
