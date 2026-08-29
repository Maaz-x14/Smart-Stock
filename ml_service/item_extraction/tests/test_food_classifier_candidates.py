"""
Phase 1 test harness: evaluate is_food classifier candidates across
Groq, Gemini, and OpenRouter.

Checks per candidate, per test item:
  (a) strict JSON-only compliance (no reasoning leakage, valid schema)
  (b) classification correctness against a small hand-labeled sample
  (c) latency

Usage:
  1. Fill in .env (see .env.example below / create alongside this script)
  2. pip install openai google-generativeai python-dotenv --break-system-packages
  3. python test_food_classifier_candidates.py

This does NOT decide anything - it only prints results per candidate for
Maaz to review and pick from. No model is hardcoded as "the" choice.
"""

import json
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# --- Prompt (locked design, do not edit without updating food_classifier.py too) ---
SYSTEM_PROMPT = """You are a binary food classifier for retail receipt items. Given an item name, output ONLY a JSON object in this exact shape:

{"is_food": true|false, "confidence": 0.0-1.0}

Rules:
- No explanation, no reasoning, no text before or after the JSON.
- "is_food" = true only for edible food/beverage items for human consumption.
- "is_food" = false for non-food items (medicine, cleaning products, toiletries, electronics, etc.).
- If genuinely ambiguous, still output your best guess with a lower confidence score - do not refuse, do not add caveats.

Examples:
Item: "ORG STRWBRY" -> {"is_food": true, "confidence": 0.95}
Item: "Supravit-M Tablet 10's" -> {"is_food": false, "confidence": 0.9}
Item: "XYZFOODS RICE" -> {"is_food": true, "confidence": 0.6}"""

# --- Hand-labeled test sample ----------------------------------------------
# SPECULATIVE / SMALL - this is not the real #34 labeled dataset, just enough
# to sanity-check candidates before committing. Replace/extend with real
# receipt items once available.
TEST_ITEMS = [
    ("ORG STRWBRY", True),
    ("WHOLE MILK", True),
    ("Supravit-M Tablet 10's", False),
    ("CHICKEN BREAST", True),
    ("Dettol Antiseptic 500ML", False),
    ("BASMATI RICE 5KG", True),
    ("Colgate Toothpaste", False),
    ("BROCCOLI", True),
    ("AA Batteries 4pk", False),
    ("SHAN MASALA MIX", True),
    ("Vaseline Lotion", False),
    ("EGGS DOZEN", True),
    ("Harpic Toilet Cleaner", False),
    ("BREAD LOAF", True),
    ("K&N'S CHICKEN NUGGETS", True),
]


@dataclass
class CandidateResult:
    name: str
    valid_json_count: int = 0
    correct_count: int = 0
    malformed_responses: list = field(default_factory=list)
    latencies_ms: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def total(self):
        return len(TEST_ITEMS)

    def summary(self):
        avg_latency = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else None
        print(f"\n=== {self.name} ===")
        print(f"Valid JSON: {self.valid_json_count}/{self.total}")
        print(f"Correct classification: {self.correct_count}/{self.total}")
        print(f"Avg latency: {avg_latency:.0f}ms" if avg_latency else "Avg latency: N/A")
        if self.malformed_responses:
            print(f"Malformed responses ({len(self.malformed_responses)}):")
            for m in self.malformed_responses:
                print(f"  - {m!r}")
        if self.errors:
            print(f"Errors ({len(self.errors)}):")
            for e in self.errors:
                print(f"  - {e}")


def parse_response(raw: str) -> dict | None:
    """Attempt to parse model output as the expected JSON shape.
    Returns None if malformed - this is deliberately strict, matching
    what food_classifier.py's real parser will need to handle."""
    try:
        cleaned = raw.strip()
        # strip common wrapper artifacts (```json fences etc.) - some
        # models add these despite instructions not to
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).strip()
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            return None
        if "is_food" not in data or "confidence" not in data:
            return None
        if not isinstance(data["is_food"], bool):
            return None
        if not isinstance(data["confidence"], (int, float)):
            return None
        if not (0.0 <= float(data["confidence"]) <= 1.0):
            return None
        return data
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def run_candidate(name: str, call_fn) -> CandidateResult:
    result = CandidateResult(name=name)
    for item_name, expected in TEST_ITEMS:
        try:
            start = time.time()
            raw = call_fn(item_name)
            elapsed_ms = (time.time() - start) * 1000
            result.latencies_ms.append(elapsed_ms)

            parsed = parse_response(raw)
            if parsed is None:
                result.malformed_responses.append(raw)
                continue

            result.valid_json_count += 1
            if parsed["is_food"] == expected:
                result.correct_count += 1
        except Exception as e:
            result.errors.append(f"{item_name!r}: {e}")
    return result


# --- Provider call functions -------------------------------------------------

def call_groq(item_name: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    resp = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Item: "{item_name}"'},
        ],
        temperature=0,
        max_tokens=100,
    )
    return resp.choices[0].message.content


def call_gemini(item_name: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT)
    resp = model.generate_content(
        f'Item: "{item_name}"',
        generation_config={"temperature": 0, "max_output_tokens": 100},
    )
    return resp.text


def call_openrouter(item_name: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Item: "{item_name}"'},
        ],
        temperature=0,
        max_tokens=100,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    candidates = []

    if os.environ.get("GROQ_API_KEY"):
        candidates.append(("Groq: " + os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"), call_groq))
    else:
        print("Skipping Groq - GROQ_API_KEY not set")

    if os.environ.get("GEMINI_API_KEY"):
        candidates.append(("Gemini: " + os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite"), call_gemini))
    else:
        print("Skipping Gemini - GEMINI_API_KEY not set")

    if os.environ.get("OPENROUTER_API_KEY"):
        # OPENROUTER_MODELS: comma-separated list, each tested as its own candidate
        models_raw = os.environ.get("OPENROUTER_MODELS", "")
        model_list = [m.strip() for m in models_raw.split(",") if m.strip()]
        if not model_list:
            print("Skipping OpenRouter - OPENROUTER_MODELS not set (comma-separated)")
        for model_id in model_list:
            candidates.append((
                f"OpenRouter: {model_id}",
                lambda item_name, m=model_id: call_openrouter(item_name, m),
            ))
    else:
        print("Skipping OpenRouter - OPENROUTER_API_KEY not set")

    results = []
    for name, fn in candidates:
        print(f"\nRunning {name}...")
        results.append(run_candidate(name, fn))

    print("\n\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for r in results:
        r.summary()
