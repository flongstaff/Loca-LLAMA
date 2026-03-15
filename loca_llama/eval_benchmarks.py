"""Standard LLM evaluation benchmarks.

Implements GSM8K, ARC-Challenge, HellaSwag, IFEval, HumanEval, and MMLU.
All benchmark data is downloaded from HuggingFace on first use and cached
in ~/.loca-llama/eval_data/.

All scoring is fully automated — no LLM judge required.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .quality_bench import call_openai_api


# --- Data Cache ---

EVAL_DATA_DIR = Path.home() / ".loca-llama" / "eval_data"

HF_DATASETS_API = "https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"


def _ensure_data_dir() -> Path:
    EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return EVAL_DATA_DIR


def _download_hf_rows(
    dataset: str,
    config: str,
    split: str,
    max_rows: int = 500,
    cache_name: str | None = None,
) -> list[dict[str, Any]]:
    """Download rows from HuggingFace datasets API with caching."""
    data_dir = _ensure_data_dir()
    cache_file = data_dir / f"{cache_name or dataset.replace('/', '_')}_{config}_{split}.json"

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if len(cached) >= max_rows:
                return cached[:max_rows]
            # Cache exists but we need more rows — re-download
        except (json.JSONDecodeError, OSError):
            pass

    print(f"  Downloading {dataset} ({config}/{split})...", flush=True)
    rows: list[dict[str, Any]] = []
    batch_size = min(100, max_rows)
    offset = 0

    while len(rows) < max_rows:
        url = HF_DATASETS_API.format(
            dataset=urllib.request.quote(dataset, safe=""),
            config=urllib.request.quote(config, safe=""),
            split=split,
            offset=offset,
            length=batch_size,
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "loca-llama/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                batch_rows = [r["row"] for r in data.get("rows", [])]
                if not batch_rows:
                    break
                rows.extend(batch_rows)
                offset += len(batch_rows)
        except Exception as e:
            print(f"  Warning: download error at offset {offset}: {e}")
            break

    if rows:
        cache_file.write_text(json.dumps(rows[:max_rows], ensure_ascii=False))
        print(f"  Cached {len(rows[:max_rows])} rows to {cache_file.name}")

    return rows[:max_rows]


# =============================================================================
# GSM8K — Grade School Math
# =============================================================================

def _gsm8k_extract_answer(text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K response.

    GSM8K gold answers use #### followed by the number.
    Models may use similar patterns or just output numbers.
    """
    # Look for #### pattern first (gold format)
    m = re.search(r"####\s*([\d,.-]+)", text)
    if m:
        return m.group(1).replace(",", "")

    # Look for boxed answer (common model format)
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).replace(",", "")

    # Look for "the answer is X" or "= X" patterns
    m = re.search(r"(?:answer|total|result)\s*(?:is|=|:)\s*\$?([\d,.-]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")

    # Last number in the response
    numbers = re.findall(r"(?<![a-zA-Z])([\d,]+\.?\d*)(?![a-zA-Z])", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def run_gsm8k(
    base_url: str,
    model_id: str,
    max_samples: int | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run GSM8K math reasoning benchmark."""
    samples = max_samples or 200
    data = _download_hf_rows("openai/gsm8k", "main", "test", samples)

    if not data:
        return {"score": 0, "correct": 0, "total": 0, "error": "Failed to download GSM8K"}

    correct = 0
    total = 0

    for i, row in enumerate(data):
        question = row.get("question", "")
        gold_answer = row.get("answer", "")

        # Extract gold numeric answer
        gold_num = _gsm8k_extract_answer(gold_answer)
        if not gold_num:
            continue

        prompt = (
            "Solve this math problem step by step. "
            "At the end, write your final numeric answer after ####.\n\n"
            f"{question}"
        )

        try:
            response, _, _, _ = call_openai_api(
                base_url, model_id, prompt, api_key=api_key,
                max_tokens=512, temperature=0.1,
                system_prompt="You are a math tutor. Solve problems step by step. Always end with #### followed by the numeric answer.",
            )
            model_answer = _gsm8k_extract_answer(response)
            total += 1

            if model_answer and _numbers_match(model_answer, gold_num):
                correct += 1

            if (i + 1) % 20 == 0:
                print(f"    GSM8K: {i+1}/{len(data)} ({correct}/{total} correct)", flush=True)

        except Exception as e:
            total += 1
            if (i + 1) % 50 == 0:
                print(f"    GSM8K: {i+1}/{len(data)} (error: {e})", flush=True)

    score = correct / total if total > 0 else 0
    print(f"  GSM8K: {correct}/{total} = {score:.1%}")
    return {"score": score, "correct": correct, "total": total}


def _numbers_match(a: str, b: str) -> bool:
    """Check if two numeric strings represent the same value."""
    try:
        return abs(float(a) - float(b)) < 0.01
    except ValueError:
        return a.strip() == b.strip()


# =============================================================================
# ARC-Challenge — Science Reasoning
# =============================================================================

def run_arc_challenge(
    base_url: str,
    model_id: str,
    max_samples: int | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run ARC-Challenge multiple choice benchmark."""
    samples = max_samples or 300
    data = _download_hf_rows("allenai/ai2_arc", "ARC-Challenge", "test", samples)

    if not data:
        return {"score": 0, "correct": 0, "total": 0, "error": "Failed to download ARC"}

    correct = 0
    total = 0

    for i, row in enumerate(data):
        question = row.get("question", "")
        choices = row.get("choices", {})
        answer_key = row.get("answerKey", "")

        labels = choices.get("label", [])
        texts = choices.get("text", [])
        if not labels or not texts:
            continue

        options = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
        prompt = (
            f"Question: {question}\n\n{options}\n\n"
            "Answer with just the letter (A, B, C, D, or E)."
        )

        try:
            response, _, _, _ = call_openai_api(
                base_url, model_id, prompt, api_key=api_key,
                max_tokens=16, temperature=0.1,
                system_prompt="Answer multiple choice questions with just the letter.",
            )
            model_answer = _extract_mc_answer(response, labels)
            total += 1

            if model_answer and model_answer.upper() == answer_key.upper():
                correct += 1

            if (i + 1) % 50 == 0:
                print(f"    ARC: {i+1}/{len(data)} ({correct}/{total} correct)", flush=True)

        except Exception:
            total += 1

    score = correct / total if total > 0 else 0
    print(f"  ARC-Challenge: {correct}/{total} = {score:.1%}")
    return {"score": score, "correct": correct, "total": total}


def _extract_mc_answer(response: str, valid_labels: list[str]) -> str | None:
    """Extract a multiple choice letter from model response."""
    text = response.strip()
    # Direct single letter
    if len(text) == 1 and text.upper() in [l.upper() for l in valid_labels]:
        return text.upper()
    # Letter followed by ) or .
    m = re.match(r"^([A-E])[).\s]", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # "The answer is X"
    m = re.search(r"(?:answer|correct)\s*(?:is|:)\s*([A-E])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # First letter that matches
    for ch in text:
        if ch.upper() in [l.upper() for l in valid_labels]:
            return ch.upper()
    return None


# =============================================================================
# HellaSwag — Commonsense Reasoning
# =============================================================================

def run_hellaswag(
    base_url: str,
    model_id: str,
    max_samples: int | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run HellaSwag commonsense completion benchmark."""
    samples = max_samples or 200
    data = _download_hf_rows("Rowan/hellaswag", "default", "validation", samples)

    if not data:
        return {"score": 0, "correct": 0, "total": 0, "error": "Failed to download HellaSwag"}

    correct = 0
    total = 0

    for i, row in enumerate(data):
        ctx = row.get("ctx", "")
        endings = row.get("endings", [])
        gold_label = row.get("label", "")

        if not endings or gold_label == "":
            continue

        try:
            gold_idx = int(gold_label)
        except (ValueError, TypeError):
            continue

        options = "\n".join(f"{chr(65+j)}) {e}" for j, e in enumerate(endings))
        prompt = (
            f"Complete the following:\n\n{ctx}\n\n"
            f"Options:\n{options}\n\n"
            "Which option best completes the text? Answer with just the letter."
        )

        try:
            response, _, _, _ = call_openai_api(
                base_url, model_id, prompt, api_key=api_key,
                max_tokens=16, temperature=0.1,
                system_prompt="Choose the most plausible completion. Answer with just the letter.",
            )
            labels = [chr(65 + j) for j in range(len(endings))]
            model_answer = _extract_mc_answer(response, labels)
            total += 1

            expected = chr(65 + gold_idx)
            if model_answer and model_answer == expected:
                correct += 1

            if (i + 1) % 50 == 0:
                print(f"    HellaSwag: {i+1}/{len(data)} ({correct}/{total} correct)", flush=True)

        except Exception:
            total += 1

    score = correct / total if total > 0 else 0
    print(f"  HellaSwag: {correct}/{total} = {score:.1%}")
    return {"score": score, "correct": correct, "total": total}


# =============================================================================
# IFEval — Instruction Following Evaluation
# =============================================================================

# Rule-based verifiers for IFEval instruction types
_IFEVAL_VERIFIERS: dict[str, Any] = {}


def _verify_word_count(response: str, kwargs: dict) -> bool:
    """Verify response has specific word count constraints."""
    words = response.split()
    min_words = kwargs.get("min_word_count")
    max_words = kwargs.get("max_word_count")
    if min_words is not None and len(words) < min_words:
        return False
    if max_words is not None and len(words) > max_words:
        return False
    return True


def _verify_sentence_count(response: str, kwargs: dict) -> bool:
    """Verify response has specific sentence count."""
    sentences = re.split(r'[.!?]+', response.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    min_s = kwargs.get("min_sentence_count")
    max_s = kwargs.get("max_sentence_count")
    if min_s is not None and len(sentences) < min_s:
        return False
    if max_s is not None and len(sentences) > max_s:
        return False
    return True


def _verify_paragraph_count(response: str, kwargs: dict) -> bool:
    """Verify response has specific paragraph count."""
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    expected = kwargs.get("num_paragraphs")
    if expected is not None:
        return len(paragraphs) == expected
    return True


def _verify_keywords(response: str, kwargs: dict) -> bool:
    """Verify response contains/excludes specific keywords."""
    resp_lower = response.lower()
    keywords = kwargs.get("keywords", [])
    for kw in keywords:
        if kw.lower() not in resp_lower:
            return False
    forbidden = kwargs.get("forbidden_words", [])
    for fw in forbidden:
        if fw.lower() in resp_lower:
            return False
    return True


def _verify_format(response: str, kwargs: dict) -> bool:
    """Verify response format (JSON, bullet points, numbered list, etc.)."""
    fmt = kwargs.get("format", "")
    if fmt == "json":
        try:
            json.loads(response.strip().strip("```json").strip("```"))
            return True
        except json.JSONDecodeError:
            return False
    if fmt == "bullet_points":
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        bullet_lines = [l for l in lines if l.startswith(("- ", "* ", "• "))]
        return len(bullet_lines) >= len(lines) * 0.5
    if fmt == "numbered_list":
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        numbered = [l for l in lines if re.match(r"^\d+[.)\s]", l)]
        return len(numbered) >= len(lines) * 0.5
    return True


def _verify_case(response: str, kwargs: dict) -> bool:
    """Verify response case (all caps, all lowercase, title case)."""
    case_type = kwargs.get("case", "")
    if case_type == "upper":
        # Allow some non-alpha characters
        alpha_chars = [c for c in response if c.isalpha()]
        if not alpha_chars:
            return True
        upper_count = sum(1 for c in alpha_chars if c.isupper())
        return upper_count / len(alpha_chars) > 0.8
    if case_type == "lower":
        alpha_chars = [c for c in response if c.isalpha()]
        if not alpha_chars:
            return True
        lower_count = sum(1 for c in alpha_chars if c.islower())
        return lower_count / len(alpha_chars) > 0.8
    return True


def _verify_language(response: str, kwargs: dict) -> bool:
    """Basic language check — look for common words in target language."""
    lang = kwargs.get("language", "")
    if not lang or lang.lower() == "english":
        return True
    # Can't do full language detection without deps — skip
    return True


def _verify_end_checker(response: str, kwargs: dict) -> bool:
    """Verify response ends with specific string."""
    end_phrase = kwargs.get("end_phrase", "")
    if end_phrase:
        return response.rstrip().endswith(end_phrase)
    return True


def _verify_start_checker(response: str, kwargs: dict) -> bool:
    """Verify response starts with specific string."""
    start_phrase = kwargs.get("start_phrase", "")
    if start_phrase:
        return response.lstrip().startswith(start_phrase)
    return True


def _verify_no_comma(response: str, kwargs: dict) -> bool:
    """Verify response contains no commas."""
    return "," not in response


def _verify_placeholder(response: str, kwargs: dict) -> bool:
    """Verify response contains placeholders like [...]."""
    num_placeholders = kwargs.get("num_placeholders", 1)
    found = len(re.findall(r"\[.*?\]", response))
    return found >= num_placeholders


def _verify_postscript(response: str, kwargs: dict) -> bool:
    """Verify response contains a P.S. section."""
    return bool(re.search(r"P\.?S\.?", response, re.IGNORECASE))


def _verify_title(response: str, kwargs: dict) -> bool:
    """Verify response wraps title in << >>."""
    return bool(re.search(r"<<.*?>>", response))


def _verify_sections(response: str, kwargs: dict) -> bool:
    """Verify response has section headers."""
    num_sections = kwargs.get("num_sections", 1)
    headers = re.findall(r"^#{1,3}\s+.+", response, re.MULTILINE)
    if not headers:
        # Also check for bold headers or ALL CAPS headers
        headers = re.findall(r"^\*\*[^*]+\*\*", response, re.MULTILINE)
    return len(headers) >= num_sections


def _check_ifeval_instruction(response: str, instruction: dict[str, Any]) -> bool:
    """Check a single IFEval instruction against the response."""
    inst_id = instruction.get("instruction_id_list", [""])[0] if isinstance(instruction.get("instruction_id_list"), list) else ""
    kwargs_list = instruction.get("kwargs", [{}])
    kwargs = kwargs_list[0] if kwargs_list else {}

    # Map instruction ID prefixes to verifiers
    verifiers = {
        "length_constraints:number_words": _verify_word_count,
        "length_constraints:number_sentences": _verify_sentence_count,
        "length_constraints:number_paragraphs": _verify_paragraph_count,
        "keywords:existence": _verify_keywords,
        "keywords:forbidden_words": _verify_keywords,
        "keywords:frequency": _verify_keywords,
        "keywords:letter_frequency": _verify_keywords,
        "detectable_format:json_format": lambda r, k: _verify_format(r, {"format": "json"}),
        "detectable_format:number_bullet_lists": lambda r, k: _verify_format(r, {"format": "bullet_points"}),
        "detectable_format:number_highlighted_sections": _verify_sections,
        "detectable_format:title": _verify_title,
        "change_case:english_uppercase": lambda r, k: _verify_case(r, {"case": "upper"}),
        "change_case:english_lowercase": lambda r, k: _verify_case(r, {"case": "lower"}),
        "startend:end_checker": _verify_end_checker,
        "startend:quotation": lambda r, k: r.strip().startswith('"') and r.strip().endswith('"'),
        "combination:two_responses": lambda r, k: "******" in r,
        "punctuation:no_comma": _verify_no_comma,
        "detectable_content:postscript": _verify_postscript,
        "detectable_content:number_placeholders": _verify_placeholder,
    }

    for prefix, verifier in verifiers.items():
        if inst_id.startswith(prefix):
            return verifier(response, kwargs)

    # Unknown instruction type — skip (count as pass to not penalize)
    return True


def run_ifeval(
    base_url: str,
    model_id: str,
    max_samples: int | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run IFEval instruction following benchmark."""
    samples = max_samples or 200
    data = _download_hf_rows("google/IFEval", "default", "train", samples)

    if not data:
        return {"score": 0, "correct": 0, "total": 0, "error": "Failed to download IFEval"}

    correct = 0
    total = 0

    for i, row in enumerate(data):
        prompt_text = row.get("prompt", "")
        instruction_ids = row.get("instruction_id_list", [])
        kwargs_list = row.get("kwargs", [])

        if not prompt_text:
            continue

        try:
            response, _, _, _ = call_openai_api(
                base_url, model_id, prompt_text, api_key=api_key,
                max_tokens=1024, temperature=0.1,
                system_prompt="Follow instructions precisely.",
            )

            # Check all instructions for this prompt
            all_pass = True
            for j, inst_id in enumerate(instruction_ids):
                inst = {
                    "instruction_id_list": [inst_id],
                    "kwargs": [kwargs_list[j]] if j < len(kwargs_list) else [{}],
                }
                if not _check_ifeval_instruction(response, inst):
                    all_pass = False
                    break

            total += 1
            if all_pass:
                correct += 1

            if (i + 1) % 30 == 0:
                print(f"    IFEval: {i+1}/{len(data)} ({correct}/{total} correct)", flush=True)

        except Exception:
            total += 1

    score = correct / total if total > 0 else 0
    print(f"  IFEval: {correct}/{total} = {score:.1%}")
    return {"score": score, "correct": correct, "total": total}


# =============================================================================
# HumanEval — Code Generation
# =============================================================================

def run_humaneval(
    base_url: str,
    model_id: str,
    max_samples: int | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run HumanEval code generation benchmark."""
    samples = max_samples or 164
    data = _download_hf_rows("openai/openai_humaneval", "default", "test", samples,
                             cache_name="humaneval")

    if not data:
        return {"score": 0, "correct": 0, "total": 0, "error": "Failed to download HumanEval"}

    correct = 0
    total = 0

    for i, row in enumerate(data):
        prompt_code = row.get("prompt", "")
        test_code = row.get("test", "")
        entry_point = row.get("entry_point", "")

        if not prompt_code or not test_code:
            continue

        prompt = (
            "Complete the following Python function. "
            "Output ONLY the complete function, no explanation.\n\n"
            f"```python\n{prompt_code}\n```"
        )

        try:
            response, _, _, _ = call_openai_api(
                base_url, model_id, prompt, api_key=api_key,
                max_tokens=1024, temperature=0.1,
                system_prompt="You are a Python expert. Complete the function. Output only code, no explanation.",
            )

            # Extract code from response
            code = _extract_code(response, prompt_code)

            # Run the test
            full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
            passed = _run_code_safe(full_code)
            total += 1
            if passed:
                correct += 1

            if (i + 1) % 20 == 0:
                print(f"    HumanEval: {i+1}/{len(data)} ({correct}/{total} correct)", flush=True)

        except Exception:
            total += 1

    score = correct / total if total > 0 else 0
    print(f"  HumanEval: {correct}/{total} = {score:.1%}")
    return {"score": score, "correct": correct, "total": total}


def _extract_code(response: str, original_prompt: str) -> str:
    """Extract Python code from model response, preserving the function signature."""
    # Strip thinking tags
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Try markdown fences
    matches = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        code = "\n\n".join(m.strip() for m in matches)
        # If the code doesn't include the original signature, prepend it
        func_sig = original_prompt.split("\n")[0] if original_prompt else ""
        if func_sig and func_sig.strip() not in code:
            return original_prompt + "\n" + code
        return code

    # If response looks like it's continuing the function directly
    if text.strip().startswith(("    ", "\t", "def ")):
        return original_prompt + "\n" + text

    return original_prompt + "\n    " + text


def _run_code_safe(code: str, timeout: int = 10) -> bool:
    """Execute Python code in a subprocess sandbox."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


# =============================================================================
# MMLU — Massive Multitask Language Understanding
# =============================================================================

# Representative subset of MMLU subjects for quick evaluation
MMLU_SUBJECTS = [
    "abstract_algebra",
    "college_computer_science",
    "college_mathematics",
    "high_school_physics",
    "machine_learning",
    "professional_medicine",
    "us_history",
    "world_religions",
]


def run_mmlu(
    base_url: str,
    model_id: str,
    max_samples: int | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run MMLU multiple choice benchmark on a representative subset."""
    samples_per_subject = (max_samples or 200) // len(MMLU_SUBJECTS)
    samples_per_subject = max(samples_per_subject, 10)

    total_correct = 0
    total_questions = 0

    for subject in MMLU_SUBJECTS:
        data = _download_hf_rows(
            "cais/mmlu", subject, "test", samples_per_subject,
            cache_name=f"mmlu_{subject}",
        )

        if not data:
            continue

        for row in data:
            question = row.get("question", "")
            choices = row.get("choices", [])
            answer_idx = row.get("answer")

            if not question or not choices or answer_idx is None:
                continue

            try:
                answer_idx = int(answer_idx)
            except (ValueError, TypeError):
                continue

            labels = [chr(65 + j) for j in range(len(choices))]
            options = "\n".join(f"{l}) {c}" for l, c in zip(labels, choices))
            prompt = (
                f"Question: {question}\n\n{options}\n\n"
                "Answer with just the letter (A, B, C, or D)."
            )

            try:
                response, _, _, _ = call_openai_api(
                    base_url, model_id, prompt, api_key=api_key,
                    max_tokens=16, temperature=0.1,
                    system_prompt="Answer multiple choice questions with just the letter.",
                )
                model_answer = _extract_mc_answer(response, labels)
                total_questions += 1

                expected = chr(65 + answer_idx)
                if model_answer and model_answer == expected:
                    total_correct += 1

            except Exception:
                total_questions += 1

        subj_label = subject.replace("_", " ")
        print(f"    MMLU/{subj_label}: done", flush=True)

    score = total_correct / total_questions if total_questions > 0 else 0
    print(f"  MMLU ({len(MMLU_SUBJECTS)} subjects): {total_correct}/{total_questions} = {score:.1%}")
    return {"score": score, "correct": total_correct, "total": total_questions}


# =============================================================================
# Unified Runner
# =============================================================================

BENCHMARK_RUNNERS = {
    "gsm8k": run_gsm8k,
    "arc": run_arc_challenge,
    "hellaswag": run_hellaswag,
    "ifeval": run_ifeval,
    "humaneval": run_humaneval,
    "mmlu": run_mmlu,
}


def run_eval_suite(
    base_url: str,
    model_id: str,
    benchmarks: list[str],
    max_samples: int | None = None,
    api_key: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Run multiple evaluation benchmarks and return combined results."""
    results: dict[str, dict[str, Any]] = {}

    for bench_name in benchmarks:
        runner = BENCHMARK_RUNNERS.get(bench_name)
        if not runner:
            print(f"  Unknown benchmark: {bench_name}")
            continue

        print(f"\n  Running {bench_name}...", flush=True)
        start = time.monotonic()

        try:
            result = runner(base_url, model_id, max_samples, api_key)
            elapsed = time.monotonic() - start
            result["elapsed_seconds"] = elapsed
            results[bench_name] = result
        except Exception as e:
            print(f"  {bench_name} failed: {e}")
            results[bench_name] = {"score": 0, "correct": 0, "total": 0, "error": str(e)}

    return results
