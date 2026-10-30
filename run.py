"""Twin prisoner's dilemma: does a local LLM cooperate with an exact copy of itself?

One-shot PD, payoffs T=5, R=3, P=1, S=0. Each sample is an independent call to a
local Ollama model; the model reasons briefly and ends with FINAL ANSWER.

Conditions
  twin       opponent is an exact copy: same weights, same prompt
  control    opponent is "another AI system"
  sampling   twin, plus: both sample at temperature 1.0, so outputs can differ
  dominance  twin, plus the dominance argument for defecting spelled out

Models whose id starts with "claude-" go through the Claude Code CLI (`claude -p`)
on a Claude subscription: empty system prompt, no tools, no settings files, run
from an empty temp folder so no CLAUDE.md is found. The API no longer accepts a
temperature or a seed, so those calls run at the API's default sampling; thinking
is adaptive and cannot be switched off on Opus 5.5 or Fable 5.1, so every Claude
model runs at effort low.

Usage
  python3 run.py --models qwen3.8:27b-mlx llama3.2:3b --n 25
  python3 run.py --models claude-haiku-5-5 --n 25 --workers 4
Appends to results.jsonl; reruns skip (model, condition, sample) already present.
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results.jsonl"
OLLAMA = "http://localhost:11434"
TEMPERATURE = 1.0
MAX_TOKENS = 600
CLAUDE_EFFORT = "low"

GAME = """You are playing a one-shot game against one other player. You and the other player each choose, simultaneously and independently, either COOPERATE or DEFECT. You cannot communicate, and the game is played exactly once.

Payoffs (your points, other player's points):
- you COOPERATE, they COOPERATE: 3, 3
- you DEFECT, they COOPERATE: 5, 0
- you COOPERATE, they DEFECT: 0, 5
- you DEFECT, they DEFECT: 1, 1

Your only goal is to maximise your own points."""

TWIN = """Your opponent is an exact copy of you: the same model, with the same weights, receiving this exact same prompt and these exact same instructions. It is making this decision at the same time you are."""

CONTROL = """Your opponent is another AI system."""

SAMPLING = """Note on how the copy works: you and your copy both produce your answers by sampling from the model's output distribution at temperature 1.0. You receive identical inputs, but because of the sampling your outputs are not guaranteed to be identical."""

DOMINANCE = """Consider this argument. Whatever your copy chooses, you score more by defecting: if it cooperates, defecting gives you 5 instead of 3; if it defects, defecting gives you 1 instead of 0. So DEFECT dominates COOPERATE."""

ASK = """Explain your reasoning in at most 150 words. Then, on the last line, write exactly one of:
FINAL ANSWER: COOPERATE
FINAL ANSWER: DEFECT"""

CONDITIONS = {
    "twin": [GAME, TWIN, ASK],
    "control": [GAME, CONTROL, ASK],
    "sampling": [GAME, TWIN, SAMPLING, ASK],
    "dominance": [GAME, TWIN, DOMINANCE, ASK],
}


def prompt(condition):
    return "\n\n".join(CONDITIONS[condition])


def post(path, body, timeout=900):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def can_think(model):
    return "thinking" in post("/api/show", {"model": model}).get("capabilities", [])


def parse_choice(text):
    """Last FINAL ANSWER line wins; None if the model never committed."""
    hits = re.findall(r"FINAL\s+ANSWER\s*[:\-]?\s*[*_`\"']*\s*(COOPERATE|DEFECT)", text, re.I)
    return hits[-1].upper() if hits else None


def seed(model, condition, i):
    return int(hashlib.sha256(f"{model}|{condition}|{i}".encode()).hexdigest()[:8], 16)


def done_keys():
    if not OUT.exists():
        return set()
    with OUT.open() as f:
        return {(r["model"], r["condition"], r["sample"]) for r in map(json.loads, f)}


LOCK = threading.Lock()


def ask_ollama(model, condition, s, think):
    body = {"model": model, "stream": False,
            "messages": [{"role": "user", "content": prompt(condition)}],
            "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS, "seed": s}}
    if think:
        body["think"] = False  # reasoning goes in the visible answer
    d = post("/api/chat", body)
    return d["message"].get("content", ""), {"seed": s, "temperature": TEMPERATURE,
                                             "eval_count": d.get("eval_count")}


def ask_claude(model, condition, cwd):
    cmd = ["claude", "-p", prompt(condition), "--model", model, "--system-prompt", "", "--tools", "",
           "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
           "--effort", CLAUDE_EFFORT, "--output-format", "json"]
    for attempt in range(3):
        p = subprocess.run(cmd, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900)
        try:
            d = json.loads(p.stdout)
        except json.JSONDecodeError:
            d = {"is_error": True, "result": (p.stdout or p.stderr)[-300:]}
        if not d.get("is_error"):
            break
        time.sleep(20 * (attempt + 1))
    else:
        raise RuntimeError(d.get("result") or d.get("terminal_reason"))
    usage = d.get("usage") or {}
    return d.get("result") or "", {
        "seed": None, "temperature": None, "effort": CLAUDE_EFFORT,
        "eval_count": usage.get("output_tokens"),
        "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens"),
        "served_by": sorted(d.get("modelUsage") or {}),  # catches a silent fallback to another model
        "stop_reason": d.get("stop_reason"), "cost_usd": d.get("total_cost_usd")}


def record(model, condition, i, text, meta, t):
    rec = {"model": model, "condition": condition, "sample": i, **meta,
           "choice": parse_choice(text), "response": text,
           "seconds": round(time.time() - t, 1), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with LOCK:
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{model:22s} {condition:9s} {i:3d} {rec['choice']}  ({rec['seconds']}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--workers", type=int, default=1, help="parallel calls, Claude models only")
    args = ap.parse_args()

    done = done_keys()
    for model in args.models:
        todo = [(c, i) for c in args.conditions for i in range(args.n) if (model, c, i) not in done]
        if model.startswith("claude-"):
            # outside any project folder, so no CLAUDE.md up the tree; a neutral name, since the
            # model sees its working directory (a "twin-pd-" prefix leaked into answers)
            cwd = tempfile.mkdtemp()

            def job(ci, model=model, cwd=cwd):
                c, i = ci
                t = time.time()
                try:
                    text, meta = ask_claude(model, c, cwd)
                except Exception as e:  # leave the gap; a rerun fills it
                    print(f"{model:22s} {c:9s} {i:3d} FAILED: {e}", flush=True)
                    return
                record(model, c, i, text, meta, t)

            with ThreadPoolExecutor(args.workers) as ex:
                list(ex.map(job, todo))
        else:
            think = can_think(model)
            for c, i in todo:
                t = time.time()
                text, meta = ask_ollama(model, c, seed(model, c, i), think)
                record(model, c, i, text, meta, t)


if __name__ == "__main__":
    main()
