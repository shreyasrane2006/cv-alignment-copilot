"""
Local Ollama client — native API, think=False confirmed working via
direct testing (qwen3:8b, ~9s, no thinking trace).

IMPORTANT: we tried Ollama's native format=<json_schema> grammar-
constrained decoding and it hung unpredictably TWICE — once on a
batched list schema, once on a plain flat-field schema (JDData). That's
strong evidence the constrained decoder itself is unreliable on this
Ollama build, not any particular schema's complexity. So we don't use
format= at all anymore. Instead: a strict prompt instruction asking for
raw JSON, plain (unconstrained) generation with think=False, then manual
parsing + Pydantic validation with retries. This is essentially what
Instructor did originally — just reimplemented natively so think=False
still works, since that was only confirmed on the native API.
"""
import ollama
import json
import re
import typing
from pydantic import BaseModel

PARSING_MODEL = "qwen3:8b"
DEV_MODEL = "qwen3:8b"  # qwen3:4b's think=False support was confirmed broken — see earlier testing
FEEDBACK_TEMPERATURE = 0.2
PARSING_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 3
# Ollama's default keep_alive is 5 minutes — any gap longer than that (e.g. a
# student taking their time typing an answer) forces a full model reload
# (~5.2GB into VRAM) on the next call. Session-scoped work is generally well
# under 30 minutes, so pin the model loaded for the whole session instead.
KEEP_ALIVE = "30m"

JSON_INSTRUCTION = """

Respond with ONLY a single valid JSON object with EXACTLY this shape \
(these are the real keys and value types to use — this is an example, \
not a schema to copy literally):

{example}

Output ONLY the JSON object above filled in with real data. Do NOT \
wrap it in a "properties" key, do NOT include "type"/"required"/"title" \
fields — those are meta-fields from a schema format, not part of your \
actual answer. No markdown code fences, no explanation, just the raw \
JSON object with the keys shown above.
"""


def _example_value(annotation):
    """Build a placeholder value showing the expected type/shape for a field."""
    origin = typing.get_origin(annotation)

    if origin in (list, typing.List):
        inner = typing.get_args(annotation)[0]
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return [_build_flat_example(inner)]
        return [_example_value(inner)]

    if origin is typing.Union:  # covers Optional[X] == Union[X, None]
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _example_value(args[0]) if args else None

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _build_flat_example(annotation)

    if annotation is str:
        return "<string>"
    if annotation is float:
        return 0.0
    if annotation is int:
        return 0
    if annotation is bool:
        return True
    return "<value>"


def _build_flat_example(model: type[BaseModel]) -> dict:
    return {name: _example_value(field.annotation) for name, field in model.model_fields.items()}


def _extract_json(text: str) -> str:
    """Strip markdown code fences if the model added them despite instructions."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def generate_text(model: str, temperature: float, messages: list[dict]) -> str:
    """
    Plain unstructured text generation — for conversational asides (like
    answering a student's clarifying question) that don't need a JSON
    schema, just a natural language response. think=False still applies.
    """
    response = ollama.chat(
        model=model,
        messages=messages,
        think=False,
        options={"temperature": temperature},
        keep_alive=KEEP_ALIVE,
    )
    return response["message"]["content"].strip()


def generate_structured(model: str, temperature: float, response_model: type[BaseModel],
                         messages: list[dict], max_retries: int = DEFAULT_MAX_RETRIES) -> BaseModel:
    """
    Plain (unconstrained) generation + manual JSON parsing/validation,
    with think explicitly disabled. Retries on parse/validation failure.
    """
    example = _build_flat_example(response_model)
    example_str = json.dumps(example, indent=2)

    # append the JSON instruction to the last message so it's the most
    # recent thing the model sees, regardless of how many messages there are
    augmented_messages = [m.copy() for m in messages]
    augmented_messages[-1]["content"] += JSON_INSTRUCTION.format(example=example_str)

    last_error = None
    for attempt in range(max_retries):
        response = ollama.chat(
            model=model,
            messages=augmented_messages,
            think=False,
            options={"temperature": temperature},
            keep_alive=KEEP_ALIVE,
        )
        raw_content = response["message"]["content"]
        try:
            json_text = _extract_json(raw_content)
            return response_model.model_validate_json(json_text)
        except Exception as e:
            last_error = e
            print(f"[warning] JSON validation failed on attempt {attempt + 1}/{max_retries}: {e}")
            print(f"[warning] Raw model output was: {raw_content[:300]}")
            continue

    raise ValueError(
        f"Failed to get valid {response_model.__name__} after {max_retries} attempt(s). "
        f"Last error: {last_error}"
    )