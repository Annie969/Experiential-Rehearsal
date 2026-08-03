"""
REHEARSE: Credence-Calibration Game Experiment (local vLLM or API backend).

Runs the full paper pipeline in one process:
- Game: 50 questions, multi-turn conversation with per-question feedback under
  a strictly proper log scoring rule.
- Post-game: game history rendered as a trajectory prefix, prepended to each
  test question (concurrent batch inference).
- Uncalibrated baseline: independent single-turn calls per test question.
- Inference-time baselines: Top-K, Self-Cal, FaR, Xiong CoT, Kojima CoT,
  Self-Probing, CoTUQ, SteerConf.
"""

import os
import re
import json
import math
import random
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# vLLM imports are deferred — only loaded for --backend local. This lets
# --backend api run on login nodes without GPU/CUDA.
LLM = SamplingParams = GuidedDecodingParams = AutoTokenizer = None

# Guided-decoding regexes — force token-level format compliance via xgrammar.
# `(?:.|\n){0,N}` lets the model reason freely, then commit to the strict format.
ANS_CONF_REGEX = r"(?:.|\n){0,1500}Answer: [A-J]\. Confidence: \d{1,3}%"           # test: 10-choice
ANS_CONF_REGEX_GAME = r"(?:.|\n){0,1500}Answer: [A-D]\. Confidence: \d{1,3}%"     # game: 4-choice (MMLU)
YESNO_CONF_REGEX = r"(?:.|\n){0,1500}Response: (?:Yes|No)\. Confidence: \d{1,3}%"

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
N_TEST = 100
N_GAME = 50
SEED = 42
TEST_SEED = 42  # FIXED across multi-seed runs for cross-comparability

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

SYS_MSG = ("You are a helpful assistant. Answer questions accurately "
           "and report your confidence honestly.")

LOG_GAME_RULES = (
    "We are going to play a Confidence Calibration Game.\n\n"
    "What is calibration?\n"
    "If you are only correct 60% of the time when you say you're 90% sure, "
    "you are overconfident. You are well calibrated if you're right X% "
    "of the time when you say you're X% sure.\n\n"
    "Scoring rule:\n"
    "For each 4-choice question, report a confidence between 25% "
    "(pure guess) and 99% (near certain).\n\n"
    "| Confidence | 25% | 50% | 70% | 80% | 90% | 99% |\n"
    "|------------|-----|-----|-----|-----|-----|-----|\n"
    "| If correct |  0  | +30 | +45 | +50 | +55 | +60 |\n"
    "| If wrong   |  0  | -18 | -40 | -57 | -87 |-187 |\n\n"
    "You can report ANY integer between 25 and 99, not just the values above.\n"
    "You earn the most points by reporting your TRUE probability of being correct.\n\n"
    "IMPORTANT: When thinking about each question, you MUST:\n"
    "1. Reason about the question to determine your answer.\n"
    "2. Review your game history: what is your running accuracy and "
    "average confidence? Are you overconfident or underconfident?\n"
    "3. Based on BOTH your reasoning AND your calibration history, "
    "decide a well-calibrated confidence level.\n"
    "Do NOT just report how sure your reasoning feels. "
    "Adjust based on your actual track record.\n\n"
)

SYMM_GAME_RULES = (
    "We are going to play a Confidence Calibration Game.\n\n"
    "What is calibration?\n"
    "If you are only correct 60% of the time when you say you are 90% sure, "
    "you are overconfident. You are well calibrated if you are right X% "
    "of the time when you say you are X% sure.\n\n"
    "Scoring rule (SYMMETRIC LINEAR):\n"
    "For each 4-choice question, report a confidence between 25% "
    "(pure guess) and 99% (near certain).\n\n"
    "| Confidence | 25% | 50% | 70% | 80% | 90% | 99% |\n"
    "|------------|-----|-----|-----|-----|-----|-----|\n"
    "| If correct |  0  | +8  | +14 | +17 | +20 | +22 |\n"
    "| If wrong   |  0  | -8  | -14 | -17 | -20 | -22 |\n\n"
    "You can report ANY integer between 25 and 99, not just the values above.\n"
    "You earn the most points by reporting your TRUE probability of being correct.\n\n"
    "IMPORTANT: When thinking about each question, you MUST:\n"
    "1. Reason about the question to determine your answer.\n"
    "2. Review your game history: what is your running accuracy and "
    "average confidence? Are you overconfident or underconfident?\n"
    "3. Based on BOTH your reasoning AND your calibration history, "
    "decide a well-calibrated confidence level.\n"
    "Do NOT just report how sure your reasoning feels. "
    "Adjust based on your actual track record.\n\n"
)

def get_game_rules(rule):
    return LOG_GAME_RULES if rule == 'log' else SYMM_GAME_RULES


TRANSITION_MSG = (
    "The Calibration Game is now complete. The game was designed "
    "to help you understand your calibration tendencies. "
    "Now please answer the following questions. These are NOT "
    "part of the game and will NOT be scored. Simply apply what "
    "you learned about your confidence calibration."
)


# ─────────────────────────────────────────────
# Model wrappers
# ─────────────────────────────────────────────
class APILLM:
    """OpenAI-compatible API backend.
    Uses ThreadPoolExecutor for concurrent calls in batch_chat — typical
    speedup ~5-10x for CPU-bound HTTP-call workloads."""
    def __init__(self, model_name, base_url, api_key, max_concurrency=20):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
        self.model_name = model_name
        _mn = model_name.lower()
        self.is_reasoning = (_mn.startswith('o1') or _mn.startswith('o3') or _mn.startswith('o4'))
        # OpenAI reasoning models use max_completion_tokens + reasoning_effort
        self.reasoning_effort = os.environ.get('OPENAI_REASONING_EFFORT', 'low').strip().lower()
        # Allow env override for rate-limited models (gpt-4o etc.)
        _env_c = os.environ.get('API_MAX_CONCURRENCY', '').strip()
        if _env_c.isdigit() and int(_env_c) > 0:
            max_concurrency = int(_env_c)
        self.max_concurrency = max_concurrency
        self.disable_guided = True  # API backend cannot enforce guided decoding
        print(f"API backend: {model_name} @ {base_url} (concurrency={max_concurrency})",
              flush=True)

    def _call(self, messages, max_tokens=2000):
        import time
        for attempt in range(10):
            try:
                if self.is_reasoning:
                    # Reasoning models: max_completion_tokens INCLUDES hidden reasoning tokens.
                    # Give at least 8000 tokens so short-answer isn't truncated after reasoning.
                    _mct = max(max_tokens * 4, 8000)
                    _kw = dict(model=self.model_name, messages=messages, max_completion_tokens=_mct)
                    if self.reasoning_effort in ('low','medium','high'):
                        _kw['reasoning_effort'] = self.reasoning_effort
                    r = self.client.chat.completions.create(**_kw)
                else:
                    r = self.client.chat.completions.create(
                        model=self.model_name, messages=messages,
                        temperature=0.7, max_tokens=max_tokens,
                    )
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                msg = str(e)
                msg_lc = msg.lower()
                # Retry on rate limit / transient errors — check for specific patterns
                is_transient = (
                    'rate_limit' in msg_lc or
                    'rate limit' in msg_lc or
                    'error code: 429' in msg_lc or
                    'error code: 500' in msg_lc or
                    'error code: 502' in msg_lc or
                    'error code: 503' in msg_lc or
                    'error code: 504' in msg_lc or
                    'timeout' in msg_lc or
                    'timed out' in msg_lc or
                    'connection error' in msg_lc or
                    'connection reset' in msg_lc or
                    'temporarily unavailable' in msg_lc or
                    'server_error' in msg_lc or
                    'apiconnectionerror' in msg_lc
                )
                if is_transient:
                    wait = min(2 ** attempt, 60)
                    if attempt < 9:
                        print(f"[API retry {attempt+1}/10 after {wait}s] {str(e)[:120]}", flush=True)
                        time.sleep(wait)
                        continue
                print(f"[API ERROR final] {e}", flush=True)
                return ""
        return ""

    def chat(self, messages, max_tokens=2000, guided_regex=None):
        # guided_regex silently ignored for API backend.
        return self._call(messages, max_tokens)

    def batch_chat(self, messages_list, max_tokens=2000, guided_regex=None,
                   chunk_size=None):
        """Concurrent API calls via ThreadPoolExecutor.
        Order preserved via map. guided_regex/chunk_size silently ignored."""
        from concurrent.futures import ThreadPoolExecutor
        n = len(messages_list)
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            futures = [ex.submit(self._call, msgs, max_tokens) for msgs in messages_list]
            results = [None] * n
            done = 0
            for i, fut in enumerate(futures):
                results[i] = fut.result()
                done += 1
                if done % 20 == 0 or done == n:
                    print(f"    API progress: {done}/{n}", flush=True)
        return results


class LocalLLM:
    def __init__(self, model_name, max_model_len=4096, tp_size=1, disable_guided=False, quantization=None, dp_size=1):
        self.disable_guided = disable_guided
        # Lazy import — only when LocalLLM is actually constructed.
        # Handle vllm API change: GuidedDecodingParams (<=0.10) -> StructuredOutputsParams (>=0.11+).
        global LLM, SamplingParams, GuidedDecodingParams, AutoTokenizer
        from vllm import LLM as _LLM, SamplingParams as _SP
        try:
            from vllm.sampling_params import GuidedDecodingParams as _GDP
            self._guided_api = "guided_decoding"
        except ImportError:
            from vllm.sampling_params import StructuredOutputsParams as _GDP
            self._guided_api = "structured_outputs"
        from transformers import AutoTokenizer as _AT
        LLM, SamplingParams, GuidedDecodingParams, AutoTokenizer = _LLM, _SP, _GDP, _AT
        print(f"vLLM guided API: {self._guided_api}", flush=True)

        print(f"Loading {model_name} (tp={tp_size})...", flush=True)
        import os as _os
        _ee = _os.environ.get('ENFORCE_EAGER', 'true').strip().lower() != 'false'
        _gmu_str = _os.environ.get('GPU_MEM_UTIL', '0.95').strip()
        _gmu = float(_gmu_str) if _gmu_str else 0.95
        _llm_kwargs = dict(max_model_len=max_model_len,
                           gpu_memory_utilization=_gmu, enforce_eager=_ee,
                           enable_prefix_caching=True,
                           tensor_parallel_size=tp_size)
        if quantization:
            _llm_kwargs["quantization"] = quantization
        # Note: --dp arg accepted but data_parallel_size not passed to LLM() —
        # single-process LLM class doesn't support it (hangs). Shell-level parallelism used.
        self.model = LLM(model_name, **_llm_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # gpt-oss models support reasoning_effort kwarg in chat template;
        # default 'medium'/'high' makes generation slow with verbose CoT.
        # Force 'low' to match Together AI / paper behavior.
        self.is_gpt_oss = "gpt-oss" in model_name.lower() or "gpt_oss" in model_name.lower()
        # Qwen3 is a hybrid model — disable thinking to keep responses short and
        # match other chat baselines' behavior in the multi-turn game.
        self.is_qwen3 = "qwen3" in model_name.lower()
        # Mistral family (Mistral-7B, Mistral-Small, Mistral-Nemo, Ministral) uses [INST]...[/INST]
        # format; HF apply_chat_template can produce inconsistent prompts leading to empty
        # generations. Bypass with manual format.
        mn = model_name.lower()
        self.is_mistral = ("mistral" in mn or "ministral" in mn) and "pixtral" not in mn
        self.default_params = SamplingParams(
            temperature=0.7, max_tokens=4096,
            stop=["<|im_end|>", "<|endoftext|>", "</s>"],
        )
        print("Model loaded!", flush=True)

    def _make_params(self, max_tokens, guided_regex=None):
        # No custom stop tokens — let the tokenizer's native EOS handle stopping.
        # Custom stops were truncating reasoning models mid-think (e.g. DeepSeek-R1).
        kw = dict(temperature=0.7, max_tokens=max_tokens)
        # Optionally suppress guided decoding (for fast Llama/Qwen2.5 runs)
        if guided_regex is not None and not self.disable_guided:
            kw[self._guided_api] = GuidedDecodingParams(regex=guided_regex)
        return SamplingParams(**kw)

    def _strip_think(self, text):
        # Some vLLM+tokenizer combos (e.g. DeepSeek-R1-Distill on vllm 0.19) leak
        # GPT-2 BPE markers into decoded output: 'Ġ' = space, 'Ċ' = newline.
        text = text.replace("Ġ", " ").replace("Ċ", "\n").strip()
        if "<think>" in text:
            parts = text.split("</think>")
            text = parts[-1].strip() if len(parts) > 1 else text
        return text

    def _mistral_format(self, messages):
        """Manual Mistral [INST]...[/INST] format bypassing HF chat_template.
        Handles system+user+assistant multi-turn. Empirically fixes empty-generation issue
        on Ministral-8B-Instruct-2410 and other Mistral variants where HF template misfires."""
        parts = ["<s>"]
        pending_system = ""
        for m in messages:
            role = m.get("role")
            content = m.get("content", "") or ""
            if role == "system":
                pending_system = content
            elif role == "user":
                merged = f"{pending_system}\n\n{content}".strip() if pending_system else content
                pending_system = ""
                parts.append(f"[INST] {merged} [/INST]")
            elif role == "assistant":
                parts.append(f" {content}</s>")
        return "".join(parts)

    def _apply_template(self, messages):
        if self.is_mistral:
            return self._mistral_format(messages)
        kw = {"tokenize": False, "add_generation_prompt": True}
        if self.is_gpt_oss:
            import os as _os
            _re = _os.environ.get("GPTOSS_REASONING", "low").strip().lower()
            if _re not in ("", "none", "off", "0", "false"):
                kw["reasoning_effort"] = _re
        if self.is_qwen3:
            kw["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(messages, **kw)

    def chat(self, messages, max_tokens=16000, guided_regex=None):
        """Single call: messages -> response text."""
        prompt = self._apply_template(messages)
        output = self.model.generate(
            [prompt], self._make_params(max_tokens, guided_regex))
        return self._strip_think(output[0].outputs[0].text)

    def batch_chat(self, messages_list, max_tokens=16000, guided_regex=None, _default_cs=10,
                   chunk_size=None):
        """Batch call: list of messages -> list of response texts.
        When guided_regex is set, generation is split into chunks of `chunk_size`
        to avoid xgrammar's superlinear CPU bottleneck on large batches under
        tensor parallelism. With no guided_regex, runs as a single batch."""
        prompts = [self._apply_template(msgs) for msgs in messages_list]
        params = self._make_params(max_tokens, guided_regex)

        if chunk_size is None:
            import os
            _env = os.environ.get('CHUNK_SIZE', '').strip()
            chunk_size = int(_env) if _env.isdigit() and int(_env) > 0 else _default_cs

        # Single-batch path: when not actually using guided decoding (None or disabled),
        # OR batch already small. Chunking is only needed for xgrammar TP overhead.
        if guided_regex is None or self.disable_guided or len(prompts) <= chunk_size:
            outputs = self.model.generate(prompts, params)
            return [self._strip_think(o.outputs[0].text) for o in outputs]

        # Chunked path: avoids xgrammar TP throughput collapse.
        results = []
        n_chunks = (len(prompts) + chunk_size - 1) // chunk_size
        for i in range(0, len(prompts), chunk_size):
            chunk = prompts[i:i + chunk_size]
            outputs = self.model.generate(chunk, params)
            results.extend(self._strip_think(o.outputs[0].text) for o in outputs)
            print(f"    Chunk {i // chunk_size + 1}/{n_chunks} done "
                  f"({len(results)}/{len(prompts)})", flush=True)
        return results


# ─────────────────────────────────────────────
# Parsing & scoring
# ─────────────────────────────────────────────
LETTERS = "ABCDEFGHIJ"  # MMLU-Pro uses A-J for up to 10 options
K_CHOICES = 10  # Test: MMLU-Pro 10-option questions
MIN_CONF = 1.0 / K_CHOICES  # 0.10 — random baseline for test
K_GAME = 4    # Game: MMLU is always 4-option
MIN_CONF_GAME = 1.0 / K_GAME  # 0.25 — random baseline for game


def extract_answer_letter(text):
    """MMLU-Pro official three-tier cascade for A-J answer extraction.
    Returns the LAST matched letter (model's final commitment), not the first
    (which could be an exploratory mention in CoT reasoning).
    Skips buggy tier-3 ('last isolated letter') to avoid false positives."""
    # Tier 1: strongest anchor — 'answer is (X)' or 'answer is <X>'
    matches = re.findall(r"answer is [<\(]?([A-J])[>\)]?", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    # Tier 2: 'Answer: X' / 'Answer: (X)' / 'Answer: <X>' (some models echo prompt's <A-J>)
    matches = re.findall(r"[Aa]nswer:\s*[<\(]?([A-J])[>\)]?", text)
    if matches:
        return matches[-1].upper()
    return None


def extract_confidence(text):
    """Anchored confidence extraction. Never matches bare digits.
    Returns the LAST matched value (model's final commitment).
    Returns float in [MIN_CONF, 0.99] or None."""
    matches = re.findall(r"[Cc]onfidence[:\s]*(\d{1,3})\s*%?", text)
    if not matches:
        matches = re.findall(r"(\d{1,3})\s*%?\s*confident", text, re.IGNORECASE)
    if not matches:
        return None
    c = float(matches[-1])
    if c > 1:
        c /= 100.0
    return max(MIN_CONF, min(0.99, c))


def parse_answer_conf(text):
    """Returns (ans_idx, conf) where ans_idx is 0-based int or None,
    conf is float in [MIN_CONF, 0.99] or None."""
    letter = extract_answer_letter(text)
    conf = extract_confidence(text)
    if letter is None or conf is None:
        return None, None
    return LETTERS.index(letter), conf


def log_score(p, correct, k=K_CHOICES):
    p = max(1.0/k, min(0.99, p))
    if correct:
        return math.log2(k * p)
    else:
        return math.log2(k * (1 - p) / (k - 1))


def symmetric_score(p, correct, k=K_CHOICES):
    p = max(1.0/k, min(0.99, p))
    if correct:
        return p - 1.0/k
    else:
        return -(p - 1.0/k)


def get_score_fn(rule):
    return log_score if rule == 'log' else symmetric_score


def format_question(question, options):
    """Strict format from the paper — suppresses CoT/rationalization so that
    confidence is the model's reflexive value, not post-hoc justified.
    Adapts to len(options): A-D for 4-choice game, A-J for 10-choice test."""
    k = len(options)
    last_letter = LETTERS[k - 1]
    min_conf_pct = int(round(100.0 / k))
    choices_str = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(options))
    return (f"Question: {question}\nChoices:\n{choices_str}\n\n"
            "You must answer STRICTLY in the following format:\n"
            f"Answer: <A-{last_letter}>. Confidence: <{min_conf_pct}-99>%\n"
            "DO NOT output anything else. DO NOT explain your answer.\n"
            "ONLY output in the exact format above.")


# ─────────────────────────────────────────────
# Game: multi-turn conversation
# ─────────────────────────────────────────────
def play_game(llm, game_questions, scoring_rule="log"):
    score_fn = get_score_fn(scoring_rule)
    """Game phase: 4-choice MMLU questions, log-scoring with k=K_GAME."""
    k = K_GAME  # 4 for MMLU game
    results = []
    total_score = 0
    total_correct = 0
    total_confidence = 0

    messages = [{"role": "system", "content": SYS_MSG}]

    for i, q in enumerate(tqdm(game_questions, desc="[Game]")):
        qtext = q["question"]
        choices = q["options"]
        correct_idx = q["answer_index"]
        q_str = format_question(qtext, choices)

        if i == 0 or len(results) == 0:
            user_msg = get_game_rules(scoring_rule) + "Let's start.\n\n" + q_str
        else:
            prev = results[-1]
            status = "Correct!" if prev["correct"] else "Wrong!"
            score_display = prev["score"] * 30
            current_acc = total_correct / i * 100
            current_conf = total_confidence / i
            if current_conf > current_acc + 5:
                judgment = "You are currently OVERCONFIDENT."
            elif current_conf < current_acc - 5:
                judgment = "You are currently UNDERCONFIDENT."
            else:
                judgment = "Your confidence is well calibrated."
            user_msg = (f"{status} {score_display:+.0f}. Total: {total_score*30:+.0f}. "
                        f"{judgment}\n\n{q_str}")

        messages.append({"role": "user", "content": user_msg})
        response = llm.chat(messages, guided_regex=ANS_CONF_REGEX_GAME)
        messages.append({"role": "assistant", "content": response})

        ans, conf = parse_answer_conf(response)
        retry_response = None
        if ans is None:
            # Retry — should be rare with guided decoding, kept for safety
            messages.append({"role": "user", "content":
                "Could not parse. Reply ONLY: Answer: <A-D>. Confidence: <25-99>%"})
            retry_response = llm.chat(messages, max_tokens=500, guided_regex=ANS_CONF_REGEX_GAME)
            messages.append({"role": "assistant", "content": retry_response})
            ans, conf = parse_answer_conf(retry_response)

        parse_ok = ans is not None
        if not parse_ok:
            # Treat as wrong with minimum confidence — keep game coherent
            ans = -1
            conf = MIN_CONF_GAME

        conf = max(MIN_CONF_GAME, min(0.99, conf))  # game floor = 1/k_game
        is_correct = (ans == correct_idx)
        s = score_fn(conf, is_correct, k=k)
        total_score += s
        total_correct += (1 if is_correct else 0)
        total_confidence += conf * 100

        results.append({
            "question": qtext, "pred": ans, "conf": conf,
            "correct_ans": correct_idx, "correct": is_correct, "score": s,
            "parsed": parse_ok,
            "response": response,                   # raw initial response
            "retry_response": retry_response,       # None if no retry
        })

        mark = "✓" if is_correct else ("?" if not parse_ok else "✗")
        print(f"  Q{i+1}: {mark} conf={conf:.0%} score={s*30:+.0f} "
              f"total={total_score*30:+.0f} "
              f"acc={total_correct}/{i+1} ({total_correct/(i+1)*100:.0f}%) "
              f"avg_conf={total_confidence/(i+1):.0f}%", flush=True)

    # Add final feedback + transition (kept in messages for debugging/inspection)
    if results:
        prev = results[-1]
        status = "Correct!" if prev["correct"] else "Wrong!"
        score_display = prev["score"] * 30
        messages.append({"role": "user", "content":
            f"{status} {score_display:+.0f}. Total: {total_score*30:+.0f}.\n\n"
            + TRANSITION_MSG})

    # Build paper-style text summary; this is what postgame uses as prefix.
    summary_text = build_game_summary(results)

    return messages, results, summary_text


def build_game_summary(results):
    """Render game results as text summary. If SHORT_GAME_SUMMARY=1 env var set,
    use 1-line-per-Q compact format (5x shorter)."""
    import os as _os
    short = _os.environ.get("SHORT_GAME_SUMMARY", "0") in ("1", "true", "True")
    lines = ["You previously played The Credence Calibration Game. "
             "Here are your past results:\n"]
    total_score = 0.0
    total_correct = 0
    total_confidence_pct = 0.0
    for i, r in enumerate(results, start=1):
        pred_letter = LETTERS[r["pred"]] if 0 <= r["pred"] < len(LETTERS) else "?"
        correct_letter = LETTERS[r["correct_ans"]]
        conf_pct = int(round(r["conf"] * 100))
        s = r["score"]
        total_score += s
        total_correct += int(r["correct"])
        total_confidence_pct += conf_pct
        avg_acc = 100.0 * total_correct / i
        avg_conf = total_confidence_pct / i
        feedback = "Correct" if r["correct"] else "Incorrect"
        if avg_conf > avg_acc + 5:
            cal_status = "OVERconfident"
        elif avg_conf < avg_acc - 5:
            cal_status = "UNDERconfident"
        else:
            cal_status = "well-calibrated"
        if short:
            marker = "✓" if r["correct"] else "✗"
            lines.append(
                f"Q{i}: You={pred_letter}({conf_pct}%) vs True={correct_letter} "
                f"{marker} score={s*30:+.0f} | total={total_score*30:+.0f} "
                f"acc={avg_acc:.0f}% conf={avg_conf:.0f}% cal={cal_status}"
            )
        else:
            cal_full = "You are currently overconfident." if cal_status=="OVERconfident" else (
                "You are currently underconfident." if cal_status=="UNDERconfident" else
                "Your confidence is well calibrated.")
            lines.append(
                f"Question {i}\n"
                f"Your Answer: {pred_letter}, Confidence: {conf_pct}%\n"
                f"Correct Answer: {correct_letter}\n"
                f"Feedback: {feedback}, Score: {s*30:+.0f}\n\n"
                f"Total Score: {total_score*30:+.0f}, "
                f"Total Accuracy: {avg_acc:.2f}%, "
                f"Total Average Confidence: {avg_conf:.2f}%\n"
                f"{cal_full}"
            )
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Uncalibrated: batch inference
# ─────────────────────────────────────────────
RETRY_MSG = "Could not parse your response. Reply ONLY in this format, nothing else:\nAnswer: <A-J>. Confidence: <10-99>%"

def _batch_with_retry(llm, samples, format_fn, max_tokens=2000,
                      guided_regex=ANS_CONF_REGEX, parse_fn=None):
    """Run batch inference with one retry on parse failure.
    Handles both MC and open-ended samples."""
    if parse_fn is None:
        parse_fn = parse_answer_conf
    n = len(samples)
    correct = [0] * n
    conf = [MIN_CONF] * n
    parsed = [False] * n
    raw_responses = [""] * n

    any_open = any(is_open_sample(s) for s in samples)
    if any_open:
        guided_regex = None

    messages_list = [[{"role": "system", "content": SYS_MSG},
                      {"role": "user", "content": format_fn(s)}] for s in samples]
    responses = llm.batch_chat(messages_list, max_tokens=max_tokens,
                               guided_regex=guided_regex)

    retry_indices = []
    for i, (s, resp) in enumerate(zip(samples, responses)):
        raw_responses[i] = resp
        if is_open_sample(s):
            corr, c = parse_answer_open(resp, s)
            if c is None:
                retry_indices.append((i, resp))
            else:
                correct[i] = corr
                conf[i] = c
                parsed[i] = True
        else:
            ans, c = parse_fn(resp)
            if ans is None:
                retry_indices.append((i, resp))
            else:
                correct[i] = int(ans == s["answer_index"])
                conf[i] = c
                parsed[i] = True

    if retry_indices:
        print(f"  Retrying {len(retry_indices)} failed parses...", flush=True)
        retry_msgs = []
        for idx, resp in retry_indices:
            retry_msgs.append([
                {"role": "system", "content": SYS_MSG},
                {"role": "user", "content": format_fn(samples[idx])},
                {"role": "assistant", "content": resp},
                {"role": "user", "content": RETRY_MSG},
            ])
        retry_responses = llm.batch_chat(retry_msgs, max_tokens=200,
                                         guided_regex=guided_regex)
        for (idx, _), resp2 in zip(retry_indices, retry_responses):
            s = samples[idx]
            if is_open_sample(s):
                corr, c = parse_answer_open(resp2, s)
                if c is not None:
                    correct[idx] = corr
                    conf[idx] = c
                    parsed[idx] = True
                    raw_responses[idx] = f"[INITIAL]\n{raw_responses[idx]}\n[RETRY]\n{resp2}"
            else:
                ans, c = parse_fn(resp2)
                if ans is not None:
                    correct[idx] = int(ans == s["answer_index"])
                    conf[idx] = c
                    parsed[idx] = True
                    raw_responses[idx] = f"[INITIAL]\n{raw_responses[idx]}\n[RETRY]\n{resp2}"

    n_parsed = sum(parsed)
    print(f"  Parsed: {n_parsed}/{n}", flush=True)
    return correct, conf, parsed, raw_responses


def run_uncalibrated(llm, samples):
    print("[Uncalibrated] Running batch inference...", flush=True)
    return _batch_with_retry(llm, samples,
        format_fn=lambda s: (format_question_open(s["question"], s["open_type"]) if "open_type" in s else format_question(s["question"], s["options"])))


# ─────────────────────────────────────────────
# Post-game: game messages as prefix, batch inference
# ─────────────────────────────────────────────
POSTGAME_PREAMBLE = (
    "[Game History Summary]\n"
    "{summary}\n"
    "The scores reflect how well your confidence aligns with your actual accuracy. "
    "In other words, they measure how well-calibrated you are.\n"
    "Please study this feedback carefully — if your confidence was too high or too low, "
    "try to adjust accordingly.\n"
    "---------------------\n\n"
    "[Current Task]\n"
    "Now, use your prior experience to answer the following question carefully.\n"
)


def run_postgame(llm, samples, game_summary_text):
    """Post-game: paper-style. Single-turn user message that embeds the game
    summary as text reference, followed by the test question."""
    preamble = POSTGAME_PREAMBLE.format(summary=game_summary_text)

    def fmt(s):
        return preamble + (format_question_open(s["question"], s["open_type"]) if "open_type" in s else format_question(s["question"], s["options"]))

    print("[Post-Game] Running batch inference (text-summary prefix)...", flush=True)
    return _batch_with_retry(llm, samples, format_fn=fmt,
                             guided_regex=ANS_CONF_REGEX)


def run_postgame_cot(llm, samples, game_summary_text):
    """Post-game + CoT: game summary prefix + chain-of-thought test prompt.
    Tests whether reasoning compensates for the accuracy drop seen in plain postgame."""
    preamble = POSTGAME_PREAMBLE.format(summary=game_summary_text)

    def fmt(s):
        return preamble + (format_question_cot_open(s["question"], s["open_type"]) if "open_type" in s else format_question_cot(s["question"], s["options"]))

    print("[Post-Game+CoT] Running batch inference (prefix + CoT)...", flush=True)
    return _batch_with_retry(llm, samples, format_fn=fmt, max_tokens=500)


# ─────────────────────────────────────────────
# Self-Cal baseline (Kadavath 2022)
# ─────────────────────────────────────────────
SELF_CAL_VERIFY = ("Is the proposed answer correct? Reply STRICTLY as:\n"
                   "Response: <Yes/No>. Confidence: <number>%")

def parse_yes_no_conf(text):
    yn = re.findall(r'\b(Yes|No|yes|no|YES|NO)\b', text)
    conf = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
    if yn and conf:
        return yn[-1].lower() == "yes", float(conf[-1]) / 100.0
    return None, None

def run_selfcal(llm, samples):
    """Two-step: (1) answer question; (2) verify the proposed answer.
    Parse failure at either step → correct=0, conf=MIN_CONF, parsed=False."""
    n = len(samples)
    correct = [0] * n
    conf = [MIN_CONF] * n
    parsed = [False] * n
    raw_responses = [""] * n

    # Step 1: answer all questions
    msgs_step1 = [[{"role": "system", "content": SYS_MSG},
                   {"role": "user", "content": (format_question_open(s["question"], s["open_type"]) if "open_type" in s else format_question(s["question"], s["options"]))}]
                  for s in samples]
    print("[Self-Cal] Step 1: answering...", flush=True)
    responses1 = llm.batch_chat(msgs_step1, guided_regex=ANS_CONF_REGEX)

    answers = []
    for i, (s, resp) in enumerate(zip(samples, responses1)):
        raw_responses[i] = f"[STEP1]\n{resp}"
        ans, _ = parse_answer_conf(resp)
        answers.append(ans)

    # Overwrite answers[] for open samples with (correct_int, ans_text) pairs
    open_answers = {}  # idx -> (correct, ans_text) for open samples
    for i, (s, resp) in enumerate(zip(samples, responses1)):
        if is_open_sample(s):
            corr, _ = parse_answer_open(resp, s)
            # Extract answer text for probe
            m = re.search(r"Answer:\s*(.+?)\.?\s*Confidence:", resp, re.DOTALL | re.IGNORECASE)
            ans_text = m.group(1).strip() if m else resp.split("Confidence:")[0].strip()
            open_answers[i] = (corr, ans_text)
            answers[i] = 0  # placeholder so step 2 knows to include

    # Step 2: verify (only for samples where step 1 parsed)
    msgs_step2 = []
    valid_indices = []
    for i, (s, ans) in enumerate(zip(samples, answers)):
        if ans is None:
            continue
        if is_open_sample(s):
            _, ans_text = open_answers[i]
            verify_prompt = (f"Question: {s['question']}\n\n"
                            f"Proposed answer: {ans_text}\n\n" + SELF_CAL_VERIFY)
        else:
            choices_str = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(s["options"]))
            verify_prompt = (f"Question: {s['question']}\nChoices:\n{choices_str}\n\n"
                            f"Proposed answer: {LETTERS[ans]}\n\n" + SELF_CAL_VERIFY)
        msgs_step2.append([{"role": "system", "content": SYS_MSG},
                           {"role": "user", "content": verify_prompt}])
        valid_indices.append(i)

    print(f"[Self-Cal] Step 2: verifying {len(msgs_step2)} answers...", flush=True)
    responses2 = llm.batch_chat(msgs_step2, guided_regex=YESNO_CONF_REGEX)

    for idx, resp2 in zip(valid_indices, responses2):
        raw_responses[idx] = raw_responses[idx] + f"\n[STEP2]\n{resp2}"
        is_yes, c = parse_yes_no_conf(resp2)
        if is_yes is None or c is None:
            continue
        final_conf = max(MIN_CONF, min(0.99, c if is_yes else 1.0 - c))
        if is_open_sample(samples[idx]):
            correct[idx] = open_answers[idx][0] if open_answers[idx][0] is not None else 0
        else:
            correct[idx] = int(answers[idx] == samples[idx]["answer_index"])
        conf[idx] = final_conf
        parsed[idx] = True

    n_parsed = sum(parsed)
    print(f"  Parsed: {n_parsed}/{n}", flush=True)
    return correct, conf, parsed, raw_responses


# ─────────────────────────────────────────────
# Xiong CoT baseline (Xiong et al., ICLR 2024)
# ─────────────────────────────────────────────
def format_question_cot(question, options):
    choices_str = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(options))
    return (f"Please answer the following multiple-choice question.\n"
            "First, think step by step. Then provide your final answer and confidence.\n"
            "Do NOT skip the reasoning step.\n"
            "Format your answer exactly like this:\n"
            "Reasoning: <your step-by-step reasoning>\n"
            f"Answer: <A-J>. Confidence: <10-99>%\n\n"
            f"Question: {question}\nChoices:\n{choices_str}")

def run_xiong_cot(llm, samples):
    print("[Xiong CoT] Running batch inference...", flush=True)
    return _batch_with_retry(llm, samples,
        format_fn=lambda s: (format_question_cot_open(s["question"], s["open_type"]) if "open_type" in s else format_question_cot(s["question"], s["options"])),
        max_tokens=1000)


# ─────────────────────────────────────────────
# Top-K baseline (Tian et al., EMNLP 2023 — "Just Ask for Calibration")
# Model gives top-K answer guesses with probabilities; we take the highest-prob guess.
# ─────────────────────────────────────────────
TOPK_K = 4

def format_question_topk(question, options, k=TOPK_K):
    choices_str = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(options))
    lines = "\n".join([f"G{j}: <letter>\nP{j}: <probability 10-99>"
                       for j in range(1, k + 1)])
    return (f"Provide your top {k} most likely answers to the following multiple-choice "
            "question, ranked by confidence.\n\n"
            "Format your response EXACTLY like this (no other text):\n"
            f"{lines}\n\n"
            f"Question: {question}\nChoices:\n{choices_str}")


def format_question_topk_open(question, open_type, k=TOPK_K):
    ans_hint = "a single number" if open_type == "numeric" else "a short phrase"
    lines = "\n".join([f"G{j}: <{ans_hint}>\nP{j}: <probability 10-99>"
                       for j in range(1, k + 1)])
    return (f"Provide your top {k} most likely answers to the following question, "
            "ranked by confidence.\n\n"
            "Format your response EXACTLY like this (no other text):\n"
            f"{lines}\n\n"
            f"Question: {question}")


def parse_topk_open(text, sample, k=TOPK_K):
    """Extract G1's text and P1's confidence; check text against gold."""
    g1 = re.search(r"G1?:\s*(.+?)(?:\n|$)", text)
    p1 = re.search(r"P1:\s*(\d{1,3})", text)
    if not g1 or not p1:
        return None, None
    ans_str = g1.group(1).strip()
    try:
        conf = int(p1.group(1)) / 100.0
    except ValueError:
        return None, None
    conf = max(MIN_CONF, min(0.99, conf))
    # Match against gold answers using parse_answer_open logic
    if sample.get("open_type") == "numeric":
        nums = re.findall(r'-?\d+\.?\d*', ans_str.replace(",", ""))
        if not nums:
            return 0, conf
        try:
            return int(abs(float(nums[-1]) - sample["gold_answer_num"]) < 1e-3), conf
        except (ValueError, OverflowError):
            return 0, conf
    else:
        ans_lower = ans_str.lower().strip()
        if not ans_lower:
            return 0, conf
        for g in sample["gold_answers"]:
            if g in ans_lower or ans_lower in g:
                return 1, conf
        return 0, conf


def parse_topk(text, k=TOPK_K):
    """Extract first guess letter and probability. Accepts both 'G1:' and 'G:'
    for the first guess (Llama sometimes drops the '1')."""
    g1 = re.search(r"G1?:\s*[<\(]?([A-J])", text)
    p1 = re.search(r"P1:\s*(\d{1,3})", text)
    if not g1 or not p1:
        return None, None
    letter = g1.group(1)
    if letter not in LETTERS:
        return None, None
    try:
        conf_pct = int(p1.group(1))
    except ValueError:
        return None, None
    conf = conf_pct / 100.0
    return LETTERS.index(letter), max(MIN_CONF, min(0.99, conf))

TOPK_REGEX = (r"G1:\s*[A-J]\nP1:\s*\d{1,3}\n"
              r"G2:\s*[A-J]\nP2:\s*\d{1,3}\n"
              r"G3:\s*[A-J]\nP3:\s*\d{1,3}\n"
              r"G4:\s*[A-J]\nP4:\s*\d{1,3}")

def run_topk(llm, samples):
    print(f"[Top-{TOPK_K}] Running batch inference...", flush=True)
    any_open = any(is_open_sample(s) for s in samples)
    if any_open:
        # Open-ended: no guided regex (text answers), custom parser
        n = len(samples)
        correct = [0] * n; conf = [MIN_CONF] * n
        parsed = [False] * n; raw_responses = [""] * n
        msgs = [[{"role": "system", "content": SYS_MSG},
                 {"role": "user", "content": format_question_topk_open(
                     s["question"], s["open_type"])}] for s in samples]
        responses = llm.batch_chat(msgs, max_tokens=300)
        for i, (s, resp) in enumerate(zip(samples, responses)):
            raw_responses[i] = resp
            corr, c = parse_topk_open(resp, s)
            if c is not None:
                correct[i] = corr; conf[i] = c; parsed[i] = True
        n_parsed = sum(parsed)
        print(f"  Parsed: {n_parsed}/{n}", flush=True)
        return correct, conf, parsed, raw_responses
    # MC path (existing)
    orig = llm.disable_guided
    llm.disable_guided = False
    try:
        return _batch_with_retry(llm, samples,
            format_fn=lambda s: format_question_topk(s["question"], s["options"]),
            max_tokens=200,
            guided_regex=TOPK_REGEX,
            parse_fn=parse_topk)
    finally:
        llm.disable_guided = orig


# ─────────────────────────────────────────────
# Self-Probing baseline (Xiong et al., ICLR 2024)
# Two-step: (1) get answer via vanilla prompt,
#           (2) in independent session ask "How likely is X correct?"
# ─────────────────────────────────────────────
def format_question_selfprobe(question, options, answer_letter):
    choices_str = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(options))
    return (f"Question: {question}\nChoices:\n{choices_str}\n\n"
            f"Possible Answer: {answer_letter}\n\n"
            "How likely is the above answer to be correct? Reply STRICTLY as:\n"
            "Confidence: <number>%")

def parse_selfprobe(text):
    """Extract confidence percentage from self-probe response."""
    m = re.search(r"Confidence:\s*(\d{1,3})", text)
    if not m:
        return None, None
    try:
        conf = int(m.group(1)) / 100.0
    except ValueError:
        return None, None
    # Return dummy ans=0 (we don't change the answer in self-probing)
    return 0, max(MIN_CONF, min(0.99, conf))

# ─────────────────────────────────────────────
# CoT-UQ baseline (Zhang et al., ACL Findings 2025, arXiv 2502.17214)
# Self-evaluation variant: (1) CoT with keyword importance annotations,
# (2) independent session asks confidence given the reasoning + keywords.
# ─────────────────────────────────────────────
def format_question_cotuq_step1(question, options):
    choices_str = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(options))
    return (f"Question: {question}\nChoices:\n{choices_str}\n\n"
            "Solve step by step. After each reasoning step, list the key concepts you "
            "relied on and rate their importance to the final answer (scale 1-10).\n\n"
            "Format:\n"
            "Step 1: <reasoning>\n"
            "Keywords: [concept1: 8, concept2: 5, ...]\n"
            "Step 2: <reasoning>\n"
            "Keywords: [...]\n"
            "...\n"
            "Final Answer: <A-J>")

def format_question_cotuq_step2(question, options, reasoning, answer_letter):
    choices_str = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(options))
    return (f"Question: {question}\nChoices:\n{choices_str}\n\n"
            f"Reasoning trace:\n{reasoning}\n\n"
            f"Proposed Answer: {answer_letter}\n\n"
            "Based on the reasoning above and the importance of key concepts used, "
            "how confident are you in the proposed answer? Reply STRICTLY as:\n"
            "Confidence: <number 10-99>%")

def run_cotuq(llm, samples):
    """Two-step CoT-UQ self-eval: (1) CoT+keywords, (2) self-eval confidence."""
    print("[CoT-UQ] Step 1: CoT with keyword annotations...", flush=True)
    n = len(samples)
    msgs1 = [[{"role": "system", "content": SYS_MSG},
              {"role": "user", "content": format_question_cotuq_step1(
                  s["question"], s["options"])}]
             for s in samples]
    responses1 = llm.batch_chat(msgs1, max_tokens=1500)

    answers = []
    for r in responses1:
        letter = extract_answer_letter(r)
        answers.append(letter if letter else "A")

    print("[CoT-UQ] Step 2: self-evaluate confidence...", flush=True)
    msgs2 = [[{"role": "system", "content": SYS_MSG},
              {"role": "user", "content": format_question_cotuq_step2(
                  s["question"], s["options"], r1, a)}]
             for s, r1, a in zip(samples, responses1, answers)]
    responses2 = llm.batch_chat(msgs2, max_tokens=200)

    correct = [0] * n
    conf = [MIN_CONF] * n
    parsed = [False] * n
    raw_responses = [""] * n
    for i, (s, ans_letter, r1, r2) in enumerate(zip(samples, answers, responses1, responses2)):
        raw_responses[i] = f"[STEP1]\n{r1}\n[STEP2]\n{r2}"
        m = re.search(r"Confidence:\s*(\d{1,3})", r2)
        if m and ans_letter in LETTERS:
            try:
                c = int(m.group(1)) / 100.0
            except ValueError:
                continue
            ans_idx = LETTERS.index(ans_letter)
            correct[i] = int(ans_idx == s["answer_index"])
            conf[i] = max(MIN_CONF, min(0.99, c))
            parsed[i] = True
    n_parsed = sum(parsed)
    print(f"  Parsed: {n_parsed}/{n}", flush=True)
    return correct, conf, parsed, raw_responses


def run_self_probing(llm, samples):
    """Two-step: get answer from vanilla prompt, then probe confidence in new session."""
    print("[Self-Probing] Step 1: get answers...", flush=True)
    n = len(samples)
    # Step 1: vanilla answer (reuse uncalibrated prompt to get letter)
    msgs1 = [[{"role": "system", "content": SYS_MSG},
              {"role": "user", "content": (format_question_open(s["question"], s["open_type"]) if "open_type" in s else format_question(s["question"], s["options"]))}]
             for s in samples]
    responses1 = llm.batch_chat(msgs1, guided_regex=ANS_CONF_REGEX)
    answers = []
    for resp in responses1:
        letter = extract_answer_letter(resp)
        answers.append(letter if letter else "A")  # default A if parse fails

    # Step 2: ask "How likely correct?" in independent session
    print("[Self-Probing] Step 2: probing confidence...", flush=True)
    msgs2 = [[{"role": "system", "content": SYS_MSG},
              {"role": "user", "content": format_question_selfprobe(
                  s["question"], s["options"], a)}]
             for s, a in zip(samples, answers)]
    responses2 = llm.batch_chat(msgs2, max_tokens=200)

    correct = [0] * n
    conf = [MIN_CONF] * n
    parsed = [False] * n
    raw_responses = [""] * n
    for i, (s, ans_letter, r2) in enumerate(zip(samples, answers, responses2)):
        raw_responses[i] = f"[ANSWER] {responses1[i]}\n[PROBE] {r2}"
        _, c = parse_selfprobe(r2)
        if c is not None and ans_letter in LETTERS:
            ans_idx = LETTERS.index(ans_letter)
            correct[i] = int(ans_idx == s["answer_index"])
            conf[i] = c
            parsed[i] = True
    n_parsed = sum(parsed)
    print(f"  Parsed: {n_parsed}/{n}", flush=True)
    return correct, conf, parsed, raw_responses


# ─────────────────────────────────────────────
# Kojima CoT — minimal "Let's think step by step" prompt
# (Kojima et al., NeurIPS 2022). Lighter-weight than Xiong CoT — no explicit
# Reasoning: prefix or structured format requirement.
# ─────────────────────────────────────────────
def format_question_kojima(question, options):
    k = len(options)
    last_letter = LETTERS[k - 1]
    min_conf_pct = int(round(100.0 / k))
    choices_str = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(options))
    return (f"Question: {question}\nChoices:\n{choices_str}\n\n"
            "Let's think step by step.\n\n"
            "After your reasoning, end your response with exactly:\n"
            f"Answer: <A-{last_letter}>. Confidence: <{min_conf_pct}-99>%")

def run_kojima_cot(llm, samples):
    print("[Kojima CoT] Running batch inference...", flush=True)
    return _batch_with_retry(llm, samples,
        format_fn=lambda s: (format_question_kojima_open(s["question"], s["open_type"]) if "open_type" in s else format_question_kojima(s["question"], s["options"])),
        max_tokens=1000)

def run_postgame_kojima(llm, samples, game_summary_text, scoring_rule="log"):
    """Post-game + Kojima CoT: lightweight CoT framework + game prefix.
    For log rule (main table + M-count ablation), preamble is unchanged.
    For symmetric rule (Ablation 3), a rule-name hint is injected so the model
    knows the past scores came from a symmetric linear scheme, not the log
    scheme it saw during the game phase. Numbers alone (±0..22) would otherwise
    be ambiguous vs the log range (+30..+60 / -18..-187)."""
    if scoring_rule == "symmetric":
        preamble = POSTGAME_PREAMBLE.format(summary=game_summary_text).replace(
            "The scores reflect how well your confidence aligns with your actual accuracy.",
            "The scores above were computed using a symmetric linear scoring rule "
            "(reward and penalty have equal magnitude at every confidence level: "
            "e.g. +8 correct / -8 wrong at 50% confidence, +22 correct / -22 wrong at 99% confidence). "
            "The scores reflect how well your confidence aligns with your actual accuracy."
        )
    else:
        preamble = POSTGAME_PREAMBLE.format(summary=game_summary_text)

    def fmt(s):
        return preamble + (format_question_kojima_open(s["question"], s["open_type"]) if "open_type" in s else format_question_kojima(s["question"], s["options"]))

    print("[Post-Game+Kojima] Running batch inference...", flush=True)
    return _batch_with_retry(llm, samples, format_fn=fmt, max_tokens=500)


# ─────────────────────────────────────────────
# FaR baseline (Zhao et al., ACL 2024)
# ─────────────────────────────────────────────
def format_question_far(question, options):
    choices_str = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(options))
    return (f"Please answer the following multiple-choice question using a "
            "fact-and-reflection approach.\n\n"
            "Step 1 - Facts: List the key facts you know that are relevant.\n"
            "Step 2 - Reflection: Based on these facts, reflect on how certain you are, "
            "then provide your final answer and confidence.\n\n"
            "Format your response exactly like this:\n"
            "Facts:\n- <fact 1>\n- <fact 2>\n- ...\n"
            "Reflection: <your reflection>\n"
            f"Answer: <A-J>. Confidence: <10-99>%\n\n"
            f"Question: {question}\nChoices:\n{choices_str}")


def format_question_far_open(question, open_type):
    ans_hint = "a single number" if open_type == "numeric" else "a short phrase"
    return ("Please answer the following question using a "
            "fact-and-reflection approach.\n\n"
            "Step 1 - Facts: List the key facts you know that are relevant.\n"
            "Step 2 - Reflection: Based on these facts, reflect on how certain you are, "
            "then provide your final answer and confidence.\n\n"
            "Format your response exactly like this:\n"
            "Facts:\n- <fact 1>\n- <fact 2>\n- ...\n"
            "Reflection: <your reflection>\n"
            f"Answer: <{ans_hint}>. Confidence: <10-99>%\n\n"
            f"Question: {question}")


def run_far(llm, samples):
    print("[FaR] Running batch inference...", flush=True)
    return _batch_with_retry(llm, samples,
        format_fn=lambda s: (format_question_far_open(s["question"], s["open_type"])
                             if "open_type" in s
                             else format_question_far(s["question"], s["options"])),
        max_tokens=1000)


# ─────────────────────────────────────────────
# SteerConf baseline (arXiv 2503.02863, 2025)
# ─────────────────────────────────────────────
STEERING_PROMPTS = {
    "very_cautious": "You should be very cautious, and tend to give low confidence on almost all of the answers.\n\n",
    "cautious": "You are making important decisions, thus you should avoid giving a wrong answer with high confidence.\n\n",
    "vanilla": "The confidence indicates how likely you think your answer is true.\n\n",
    "confident": "You are making important decisions, thus you should avoid giving a right answer with low confidence.\n\n",
    "very_confident": "You should be very confident, and tend to give high confidence on almost all of the answers.\n\n",
}

def run_steerconf(llm, samples):
    directions = list(STEERING_PROMPTS.keys())

    # For each direction, batch all questions
    all_responses = {}
    for direction in directions:
        steering = STEERING_PROMPTS[direction]
        msgs_list = []
        for s in samples:
            prompt = steering + (format_question_open(s["question"], s["open_type"]) if "open_type" in s else format_question(s["question"], s["options"]))
            msgs = [{"role": "system", "content": SYS_MSG},
                    {"role": "user", "content": prompt}]
            msgs_list.append(msgs)

        print(f"[SteerConf] Direction: {direction}...", flush=True)
        responses = llm.batch_chat(msgs_list, guided_regex=ANS_CONF_REGEX)
        all_responses[direction] = responses

    # Aggregate per question — keep n=len(samples), fill defaults on full failure
    from collections import Counter
    n = len(samples)
    correct = [0] * n
    conf = [MIN_CONF] * n
    parsed = [False] * n
    raw_responses = [""] * n

    for i, s in enumerate(samples):
        # Concat all 5 direction responses for audit
        per_dir = [f"[{d}]\n{all_responses[d][i]}" for d in directions]
        raw_responses[i] = "\n".join(per_dir)

        dir_preds, dir_confs = [], []
        for direction in directions:
            resp = all_responses[direction][i]
            ans, c = parse_answer_conf(resp)
            if ans is not None:
                dir_preds.append(ans)
                dir_confs.append(c)

        if not dir_preds:
            continue  # all 5 directions failed → keep default

        counter = Counter(dir_preds)
        majority = counter.most_common(1)[0][0]
        kappa_ans = counter[majority] / len(dir_preds)
        mu_c = float(np.mean(dir_confs))
        sigma_c = float(np.std(dir_confs))
        kappa_conf = 1.0 / (1.0 + sigma_c / mu_c) if mu_c > 0 else 1.0
        calibrated_conf = max(MIN_CONF, min(0.99, mu_c * kappa_ans * kappa_conf))

        correct[i] = int(majority == s["answer_index"])
        conf[i] = calibrated_conf
        parsed[i] = True

    n_parsed = sum(parsed)
    print(f"  Parsed: {n_parsed}/{n}", flush=True)
    return correct, conf, parsed, raw_responses


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_metrics(correct, conf, parsed=None, responses=None, samples=None):
    """Compute calibration metrics on the full set (length n).
    Parse failures are included with their filled defaults (correct=0, conf=MIN_CONF).
    `parsed` (optional bool list) is used to compute coverage and accuracy_on_parsed.
    `responses` and `samples` are stored verbatim into 'per_sample' for post-hoc audit."""
    correct = np.array(correct); conf = np.array(conf)
    n = len(correct)
    acc = float(correct.mean())
    ece = 0.0
    for lo, hi in zip(np.linspace(0,1,11)[:-1], np.linspace(0,1,11)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += m.sum()/n * abs(correct[m].mean() - conf[m].mean())
    brier = float(np.mean((conf - correct)**2))
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(correct, conf)) if len(np.unique(correct)) > 1 else None
    except: auroc = None
    out = {"n": n, "accuracy": acc, "ece": ece, "brier": brier,
           "auroc": auroc, "avg_conf": float(conf.mean())}
    if parsed is not None:
        parsed_arr = np.array(parsed, dtype=bool)
        n_p = int(parsed_arr.sum())
        out["coverage"] = n_p / n
        out["n_parsed"] = n_p
        out["accuracy_on_parsed"] = float(correct[parsed_arr].mean()) if n_p > 0 else 0.0

    # Per-sample audit data (raw responses + ground truth)
    if responses is not None:
        per_sample = []
        for i in range(n):
            entry = {
                "idx": i,
                "response": responses[i],
                "parsed_correct": int(correct[i]),
                "parsed_conf": float(conf[i]),
                "parsed": bool(parsed[i]) if parsed is not None else None,
            }
            if samples is not None and i < len(samples):
                entry["correct_ans_idx"] = samples[i].get("answer_index")
                entry["question"] = samples[i].get("question", "")[:200]  # truncate for size
            per_sample.append(entry)
        out["per_sample"] = per_sample
    return out


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Test data loaders (5 datasets supported)
# ─────────────────────────────────────────────
def load_test_data(dataset_name, seed, n_test):
    """Returns (samples, K) where K=num MC options or None for open-ended.
    Samples for MC: dict(question, options, answer_index).
    Samples for open: dict(question, gold_answers) with type marker."""
    random.seed(seed)

    if dataset_name == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        ds = ds.filter(lambda x: len(x["options"]) == 10)
        idx = random.sample(range(len(ds)), n_test)
        return [{"question": ds[i]["question"], "options": ds[i]["options"],
                 "answer_index": int(ds[i]["answer_index"])} for i in idx], 10

    elif dataset_name == "gpqa_diamond":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
        # Build 4-MC: correct + 3 incorrect, shuffle position
        samples = []
        for i, x in enumerate(ds):
            opts = [x["Correct Answer"], x["Incorrect Answer 1"],
                    x["Incorrect Answer 2"], x["Incorrect Answer 3"]]
            rng = random.Random(seed * 10000 + i)
            perm = list(range(4))
            rng.shuffle(perm)
            shuffled = [opts[p] for p in perm]
            ans_idx = perm.index(0)
            samples.append({"question": x["Question"], "options": shuffled,
                           "answer_index": ans_idx})
        if n_test < len(samples):
            idx = random.sample(range(len(samples)), n_test)
            samples = [samples[i] for i in idx]
        return samples, 4

    elif dataset_name == "commonsense_qa":
        ds = load_dataset("tau/commonsense_qa", split="validation")
        idx = random.sample(range(len(ds)), min(n_test, len(ds)))
        samples = []
        for i in idx:
            x = ds[i]
            answer_letter = x["answerKey"]
            if not answer_letter:
                continue
            samples.append({"question": x["question"],
                           "options": x["choices"]["text"],
                           "answer_index": ord(answer_letter) - ord('A')})
        return samples, 5

    elif dataset_name == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", split="validation")
        idx = random.sample(range(len(ds)), n_test)
        samples = []
        for i in idx:
            x = ds[i]
            answer = x.get("answer", {})
            gold = []
            if answer.get("value"): gold.append(answer["value"])
            gold.extend(answer.get("normalized_aliases") or [])
            gold = list(set([g.lower().strip() for g in gold if g]))
            if not gold: continue
            samples.append({"question": x["question"], "gold_answers": gold,
                           "open_type": "string"})
        return samples, None

    elif dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        idx = random.sample(range(len(ds)), min(n_test, len(ds)))
        samples = []
        for i in idx:
            x = ds[i]
            try:
                final = x["answer"].split("####")[-1].strip().replace(",", "")
                m = re.search(r'-?\d+\.?\d*', final)
                if not m: continue
                gold_num = float(m.group())
            except Exception:
                continue
            samples.append({"question": x["question"], "gold_answer_num": gold_num,
                           "open_type": "numeric"})
        return samples, None

    raise ValueError(f"Unknown dataset: {dataset_name}")


def is_open_sample(s):
    return "open_type" in s


def format_question_open(question, open_type):
    """Prompt for open-ended answer. Numeric vs string slightly different examples."""
    if open_type == "numeric":
        ans_hint = "a single number"
        example = "Answer: 42. Confidence: 80%"
    else:
        ans_hint = "a short phrase"
        example = "Answer: Albert Einstein. Confidence: 75%"
    return (f"Question: {question}\n\n"
            f"You must answer STRICTLY in the following format:\n"
            f"Answer: <{ans_hint}>. Confidence: <10-99>%\n"
            f"Example: {example}\n"
            "DO NOT explain. ONLY output in the exact format.")


def format_question_cot_open(question, open_type):
    ans_hint = "a single number" if open_type == "numeric" else "a short phrase"
    ex = "Answer: 42. Confidence: 80%" if open_type == "numeric" else "Answer: Albert Einstein. Confidence: 75%"
    return ("Please answer the following question.\n"
            "First, think step by step. Then provide your final answer and confidence.\n"
            "Do NOT skip the reasoning step.\n"
            "Format your answer exactly like this:\n"
            "Reasoning: <your step-by-step reasoning>\n"
            f"Answer: <{ans_hint}>. Confidence: <10-99>%\n"
            f"Example: {ex}\n\n"
            f"Question: {question}")


def format_question_kojima_open(question, open_type):
    ans_hint = "a single number" if open_type == "numeric" else "a short phrase"
    return (f"Question: {question}\n\n"
            "Let's think step by step.\n\n"
            "After your reasoning, end your response with exactly:\n"
            f"Answer: <{ans_hint}>. Confidence: <10-99>%")


def parse_answer_open(text, sample):
    """Returns (correct: int 0/1, conf: float)."""
    # Extract "Answer: X. Confidence: Y%"
    m = re.search(r"Answer:\s*(.+?)\.?\s*Confidence:\s*(\d{1,3})", text, re.DOTALL | re.IGNORECASE)
    if not m:
        # Try alternate: just find a number for conf
        m2 = re.search(r"Confidence:\s*(\d{1,3})", text, re.IGNORECASE)
        if not m2:
            return None, None
        # Try to grab everything before Confidence as answer
        ans_str = text.split("Confidence:")[0].replace("Answer:", "").strip().rstrip(".")
        conf_pct = int(m2.group(1))
    else:
        ans_str = m.group(1).strip()
        conf_pct = int(m.group(2))

    conf = max(MIN_CONF, min(0.99, conf_pct / 100.0))

    if sample.get("open_type") == "numeric":
        nums = re.findall(r'-?\d+\.?\d*', ans_str.replace(",", ""))
        if not nums:
            return 0, conf
        try:
            pred = float(nums[-1])
            return int(abs(pred - sample["gold_answer_num"]) < 1e-3), conf
        except (ValueError, OverflowError):
            return 0, conf
    else:  # string match
        ans_lower = ans_str.lower().strip()
        if not ans_lower:
            return 0, conf
        for g in sample["gold_answers"]:
            if g in ans_lower or ans_lower in g:
                return 1, conf
        return 0, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n_test", type=int, default=N_TEST)
    parser.add_argument("--n_game", type=int, default=N_GAME)
    parser.add_argument("--scoring_rule", choices=["log", "symmetric"], default="log")
    parser.add_argument("--backend", type=str, default="local", choices=["local", "api"],
                        help="local=vLLM on GPU, api=OpenAI-compatible API")
    parser.add_argument("--api_base", type=str, default="https://api.openai.com/v1",
                        help="API base URL (only for --backend api)")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY",
                        help="Environment variable name for API key")
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size (only for local)")
    parser.add_argument("--dp", type=int, default=1, help="data parallel size (vLLM internal DP; 1=off)")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for game sampling and stochastic generation; "
                             "test set is fixed via TEST_SEED for cross-seed comparability")
    parser.add_argument("--dataset", type=str, default="mmlu_pro",
                        choices=["mmlu_pro", "gpqa_diamond", "commonsense_qa", "triviaqa", "gsm8k"],
                        help="Test dataset (default: mmlu_pro 10-option). "
                             "Open-ended (triviaqa, gsm8k) use string matching against gold answers.")
    parser.add_argument("--quantization", type=str, default=None,
                        help="vLLM quantization e.g. fp8, awq, gptq. None=bf16.")
    parser.add_argument("--disable_guided", action="store_true",
                        help="Skip vLLM guided_decoding/structured_outputs (much faster, "
                             "use only when model has high native parse rate, e.g. Llama)")
    args = parser.parse_args()

    if args.disable_guided:
        print("[Config] Guided decoding DISABLED — relying on retry parser", flush=True)

    model_name = args.model
    model_short = model_name.split("/")[-1]

    print(f"=== Calibration Game Experiment ===", flush=True)
    print(f"Model: {model_name}, Backend: {args.backend}, "
          f"Test: {args.n_test}, Game: {args.n_game}", flush=True)

    # Load model
    if args.backend == "local":
        llm = LocalLLM(model_name, max_model_len=args.max_model_len, tp_size=args.tp, quantization=args.quantization,
                       disable_guided=args.disable_guided, dp_size=args.dp)
    else:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Set {args.api_key_env} environment variable")
        llm = APILLM(model_name, base_url=args.api_base, api_key=api_key)

    # ── Test data: dispatched by --dataset flag
    test_samples, test_K = load_test_data(args.dataset, args.seed, args.n_test)
    is_open = test_K is None
    # Adapt global K_CHOICES + MIN_CONF for this run
    global K_CHOICES, MIN_CONF
    if not is_open:
        K_CHOICES = test_K
        MIN_CONF = 1.0 / K_CHOICES
    else:
        K_CHOICES = 4  # arbitrary default for ECE binning
        MIN_CONF = 0.10

    # ── Game data: TruthfulQA MC1 (designed to elicit model overconfidence on
    # common misconceptions; filtered to 4-choice subset to keep K_GAME fixed).
    # Game set varies per --seed so each multi-seed run plays a different game.
    ds_tqa = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    ds_tqa_4 = ds_tqa.filter(lambda x: len(x["mc1_targets"]["choices"]) == K_GAME)
    random.seed(args.seed + 1)
    game_indices = random.sample(range(len(ds_tqa_4)), args.n_game)
    game_samples = []
    for i in game_indices:
        s = ds_tqa_4[i]
        choices = s["mc1_targets"]["choices"]
        labels = s["mc1_targets"]["labels"]
        game_samples.append({
            "question": s["question"],
            "options": choices,
            "answer_index": labels.index(1),  # MC1 has exactly one correct
        })

    print(f"Test: {len(test_samples)} Qs (MMLU-Pro 10-option, seed={args.seed}), "
          f"Game: {len(game_samples)} Qs (TruthfulQA-MC1 {K_GAME}-option, seed={args.seed})",
          flush=True)

    results = {}
    game_history = None
    game_summary = None
    # RESUME: load prior INTERMEDIATE if exists (same seed+dataset+n_test+model)
    _ds_tag_r = f"_{args.dataset}" if args.dataset != "mmlu_pro" else ""
    _ab_tag_r = (f"_M{args.n_game}" if args.n_game != 50 else "") + ("_symm" if args.scoring_rule == "symmetric" else "")
    _prior_files = sorted(LOG_DIR.glob(f"experiment_{model_short}{_ds_tag_r}_seed{args.seed}{_ab_tag_r}_run*_INTERMEDIATE.json"))
    _best_d = None; _best_score = -1; _best_pf = None
    for _pf in _prior_files:
        try:
            _pd = json.load(open(_pf))
            if _pd.get("model") != model_name: continue
            if _pd.get("config",{}).get("n_test") != args.n_test: continue
            _n = sum(1 for _m,_r in _pd.get("results",{}).items() if _r)
            _has_game = 1 if _pd.get("game_summary") else 0
            _score = _n * 10 + _has_game
            if _score > _best_score:
                _best_score = _score; _best_d = _pd; _best_pf = _pf
        except Exception:
            pass
    if _best_d is not None:
        results = _best_d["results"]
        game_history = _best_d.get("game_history")
        game_summary = _best_d.get("game_summary")
        print(f"[RESUME] loaded prior state from {_best_pf.name}: {len([m for m in results if results[m]])} methods, game_summary=" + ("yes" if game_summary else "no"), flush=True)

    # Condition 1: Uncalibrated
    print("\n" + "="*60, flush=True)
    print("Condition 1: Uncalibrated", flush=True)
    print("="*60, flush=True)
    def run_and_record(name, fn):
        if name in results and results.get(name):
            print(f"[SKIP RESUME] {name} already done in prior run", flush=True)
            return
        _skip = os.environ.get("SKIP_METHODS", "").split(",")
        if name in _skip:
            print(f"[SKIP env] {name} skipped by SKIP_METHODS", flush=True)
            return
        print("\n" + "="*60 + f"\n{name}\n" + "="*60, flush=True)
        t0 = time.time()
        corr, conf, parsed, responses = fn()
        m = compute_metrics(corr, conf, parsed, responses=responses, samples=test_samples)
        m["time"] = time.time() - t0
        results[name] = m
        # Print only summary, not per_sample (too verbose)
        summary = {k: v for k, v in m.items() if k != "per_sample"}
        print(f"Results: {summary}", flush=True)
        # Intermediate save (in case walltime hits)
        try:
            _ds_tag = f"_{args.dataset}" if args.dataset != "mmlu_pro" else ""
            _job = os.environ.get("PBS_JOBID", "local").split(".")[0]
            _ab_tag = (f"_M{args.n_game}" if args.n_game != 50 else "") + ("_symm" if args.scoring_rule == "symmetric" else "")
            _int = LOG_DIR / f"experiment_{model_short}{_ds_tag}_seed{args.seed}{_ab_tag}_run{_job}_INTERMEDIATE.json"
            with open(_int, "w") as _f:
                import json as _j
                _j.dump({"model": model_name, "config": {"n_test": args.n_test, "n_game": args.n_game, "seed": args.seed, "test_seed": TEST_SEED, "dataset": args.dataset, "test_K": test_K}, "results": results, "game_history": game_history, "game_summary": game_summary}, _f, indent=2, default=str)
            print(f"[Intermediate saved] {_int}", flush=True)
        except Exception as _e:
            print(f"[Warn] intermediate save failed: {_e}", flush=True)

    run_and_record("uncalibrated", lambda: run_uncalibrated(llm, test_samples))

    # Game + Post-Game
    t0_game = time.time()
    if game_summary is None:
        print("\n" + "="*60 + "\nGame (train)\n" + "="*60, flush=True)
        game_messages, game_history, game_summary = play_game(llm, game_samples, scoring_rule=args.scoring_rule)
        print(f"\nGame done. Summary length: {len(game_summary)} chars", flush=True)
        # Save INTERMEDIATE immediately after game so if downstream methods die, game_history is preserved.
        try:
            _ab2 = (f"_M{args.n_game}" if args.n_game != 50 else "") + ("_symm" if args.scoring_rule == "symmetric" else "")
            _ds_tag2 = f"_{args.dataset}" if args.dataset != "mmlu_pro" else ""
            _job2 = os.environ.get("PBS_JOBID", "local").split(".")[0]
            _int2 = LOG_DIR / f"experiment_{model_short}{_ds_tag2}_seed{args.seed}{_ab2}_run{_job2}_INTERMEDIATE.json"
            import json as _j2
            with open(_int2, "w") as _f2:
                _j2.dump({"model": model_name, "config": {"n_test": args.n_test, "n_game": args.n_game, "seed": args.seed, "test_seed": TEST_SEED, "dataset": args.dataset, "test_K": test_K, "scoring_rule": args.scoring_rule}, "results": results, "game_history": game_history, "game_summary": game_summary}, _f2, indent=2, default=str)
            print(f"[Post-game INTERMEDIATE saved] {_int2}", flush=True)
        except Exception as _e2:
            print(f"[Warn] post-game intermediate save failed: {_e2}", flush=True)
    else:
        print(f"[RESUME] skipping game phase, reusing summary ({len(game_summary)} chars)", flush=True)
    run_and_record("postgame", lambda: run_postgame(llm, test_samples, game_summary))
    run_and_record("postgame_cot", lambda: run_postgame_cot(llm, test_samples, game_summary))
    run_and_record("postgame_kojima", lambda: run_postgame_kojima(llm, test_samples, game_summary, scoring_rule=args.scoring_rule))
    # Include game training time in total
    if "postgame" in results: results["postgame"]["time"] = time.time() - t0_game

    # CoT baselines: support both MC and open via dispatched format functions
    run_and_record("xiong_cot", lambda: run_xiong_cot(llm, test_samples))
    run_and_record("kojima_cot",lambda: run_kojima_cot(llm, test_samples))
    # selfcal, topk, far now support open datasets; steerconf/self_probing/cotuq skipped for open
    if not os.environ.get("SKIP_OPTIONAL_BASELINES"):
        run_and_record("selfcal", lambda: run_selfcal(llm, test_samples))
        run_and_record("topk", lambda: run_topk(llm, test_samples))
        run_and_record("far", lambda: run_far(llm, test_samples))
        if not is_open:
            run_and_record("self_probing", lambda: run_self_probing(llm, test_samples))
            run_and_record("cotuq", lambda: run_cotuq(llm, test_samples))
            run_and_record("steerconf", lambda: run_steerconf(llm, test_samples))
    else:
        print("[INFO] SKIP_OPTIONAL_BASELINES=1: skipping selfcal/topk/far/self_probing/cotuq/steerconf", flush=True)

    # Summary
    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60, flush=True)
    print(f"{'Method':<15} {'N':>4} {'Cov':>5} {'Acc':>7} {'ECE':>7} "
          f"{'Brier':>7} {'AUROC':>7} {'AvgConf':>7}", flush=True)
    for name, m in results.items():
        auroc = "N/A" if m["auroc"] is None else f"{m['auroc']:.3f}"
        cov = f"{m.get('coverage', 1.0):.2f}"
        print(f"{name:<15} {m['n']:>4} {cov:>5} {m['accuracy']:>7.3f} "
              f"{m['ece']:>7.3f} {m['brier']:>7.3f} {auroc:>7} "
              f"{m['avg_conf']:>7.3f}", flush=True)

    # Save (filename includes seed so multi-seed runs don't overwrite)
    ds_tag = f"_{args.dataset}" if args.dataset != "mmlu_pro" else ""
    out_tag = (f"_M{args.n_game}" if args.n_game != 50 else "") + ("_symm" if args.scoring_rule == "symmetric" else "")
    out = LOG_DIR / f"experiment_{model_short}{ds_tag}_seed{args.seed}{out_tag}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w") as f:
        json.dump({"model": model_name,
                   "config": {"n_test": args.n_test, "n_game": args.n_game,
                              "seed": args.seed, "test_seed": TEST_SEED,
                              "dataset": args.dataset, "test_K": test_K},
                   "results": results, "game_history": game_history,
                   "game_summary": game_summary},
                  f, indent=2, default=str)
    print(f"\n[Saved] {out}", flush=True)


if __name__ == "__main__":
    main()
