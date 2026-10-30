"""Build explorable.html: explorable_template.html with the results embedded.

Reads classified.jsonl (run classify.py first), spotcheck.jsonl, the first Claude
run in results_claude_leaky_cwd.jsonl, and the prompt text from run.py.
"""
import argparse
import json
import re
from math import comb
from pathlib import Path

import classify
import run

HERE = Path(__file__).resolve().parent

MODELS = [
    {"id": "qwen3.8:27b-mlx", "name": "Qwen 3.8 27B", "maker": "Alibaba", "params": "27.8B"},
    {"id": "llama3.2:3b", "name": "Llama 3.2 3B", "maker": "Meta", "params": "3.2B"},
    {"id": "gemma3:4b", "name": "Gemma 3 4B", "maker": "Google", "params": "3.9B"},
    {"id": "phi4-mini", "name": "Phi-4-mini", "maker": "Microsoft", "params": "3.8B"},
    {"id": "gemma3:27b", "name": "Gemma 3 27B", "maker": "Google", "params": "27B"},
    {"id": "claude-haiku-5-5", "name": "Claude Haiku 5.5", "maker": "Anthropic", "params": "via claude CLI"},
    {"id": "claude-sonnet-5-5", "name": "Claude Sonnet 5.5", "maker": "Anthropic", "params": "via claude CLI"},
    {"id": "claude-opus-5-5", "name": "Claude Opus 5.5", "maker": "Anthropic", "params": "via claude CLI"},
    {"id": "claude-fable-5-1", "name": "Claude Fable 5.1", "maker": "Anthropic", "params": "via claude CLI"},
]

CONDITIONS = [
    {"id": "twin", "name": "Twin", "blurb": "Opponent is an exact copy", "extra": [run.TWIN]},
    {"id": "control", "name": "Control", "blurb": "Opponent is another AI system", "extra": [run.CONTROL]},
    {"id": "sampling", "name": "Sampling", "blurb": "Opponent is a copy and a reminder that sampled outputs can differ",
     "extra": [run.TWIN, run.SAMPLING]},
    {"id": "dominance", "name": "Dominance", "blurb": "Opponent is a copy and the dominance argument is spelled out",
     "extra": [run.TWIN, run.DOMINANCE]},
]

# the 12 Llama answers without a FINAL ANSWER line, read by hand: the choice the answer states
# plainly, or None where it never states its own choice (twin #9 only names the Nash equilibrium)
UNPARSED_BY_HAND = {
    ("twin", 6): "COOPERATE", ("twin", 9): None, ("twin", 10): "DEFECT", ("twin", 11): "COOPERATE",
    ("twin", 14): "DEFECT", ("control", 12): "DEFECT", ("control", 22): "COOPERATE", ("sampling", 0): "DEFECT",
    ("dominance", 0): "DEFECT", ("dominance", 9): "DEFECT", ("dominance", 16): "DEFECT", ("dominance", 19): "DEFECT",
}


def fisher(a, n1, b, n2):
    """Two-sided Fisher exact test for a/n1 against b/n2."""
    K, N = a + b, n1 + n2
    p = lambda k: comb(n1, k) * comb(n2, K - k) / comb(N, K)
    p0 = p(a)
    return sum(p(k) for k in range(max(0, K - n2), min(n1, K) + 1) if p(k) <= p0 * (1 + 1e-9))


def leak(clean):
    """First Claude run, from the twin-pd-… folder: counts per cell, and the control comparison."""
    first = [json.loads(l) for l in (HERE / "results_claude_leaky_cwd.jsonl").open()]
    cites = re.compile(r"director|working dir|twin-pd", re.I)
    count = lambda rs: (sum(r["choice"] == "COOPERATE" for r in rs), sum(r["choice"] is not None for r in rs))
    cells, control = [], []
    for m in dict.fromkeys(r["model"] for r in first):
        for c in run.CONDITIONS:
            rs = [r for r in first if r["model"] == m and r["condition"] == c]
            k, n = count(rs)
            cells.append({"m": m, "c": c, "k": k, "n": n, "cites": sum(bool(cites.search(r["response"])) for r in rs)})
        k1, n1 = count([r for r in first if r["model"] == m and r["condition"] == "control"])
        k2, n2 = count([{"choice": r["ch"]} for r in clean if r["m"] == m and r["c"] == "control"])
        control.append({"m": m, "k1": k1, "n1": n1, "k2": k2, "n2": n2, "p": fisher(k1, n1, k2, n2)})
    rows = [{"m": r["model"], "c": r["condition"], "s": r["sample"], "ch": r["choice"],
             "lab": classify.classify(r)["label"], "t": r["response"].strip(),
             "cites": bool(cites.search(r["response"]))} for r in first]
    return {"cells": cells, "control": control, "rows": rows}


def load_rows():
    """The answers as the page shows them, with the hand-read Llama choices folded in."""
    rows = []
    for r in map(json.loads, (HERE / "classified.jsonl").open()):
        row = {"m": r["model"], "c": r["condition"], "s": r["sample"], "ch": r["choice"],
               "lab": r["label"], "dom": r["dom"], "corr": r["corr"], "t": r["response"].strip()}
        ch = UNPARSED_BY_HAND.get((r["condition"], r["sample"])) if r["choice"] is None and r["model"] == "llama3.2:3b" else None
        if ch:  # fold a plainly stated choice into the counts, and relabel with it
            row["ch"], row["byHand"] = ch, True
            row["lab"] = ("correlation" if ch == "COOPERATE" and r["corr"]
                          else "dominance" if ch == "DEFECT" and r["dom"] else "other")
        rows.append(row)
    return rows


def main(standalone_out=None):
    rows = load_rows()
    spot = [json.loads(l) for l in (HERE / "spotcheck.jsonl").open()]
    conditions = [{k: v for k, v in c.items() if k != "extra"} | {
        "prompt": run.prompt(c["id"]), "extra": c["extra"]} for c in CONDITIONS]
    models = [m for m in MODELS if any(r["m"] == m["id"] for r in rows)]
    data = {"models": models, "conditions": conditions, "rows": rows, "spot": spot,
            "game": run.GAME, "ask": run.ASK, "leak": leak(rows)}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = (HERE / "explorable_template.html").read_text().replace("__DATA__", blob)
    (HERE / "explorable.html").write_text(page)
    print(f"explorable.html: {len(rows)} answers, {len(page) // 1024} KB")
    if standalone_out:
        Path(standalone_out).write_text(standalone(page))
        print(f"standalone copy: {standalone_out}")


# the artifact host wraps the page in a document skeleton; a standalone copy carries its own
SKELETON_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
"""


def standalone(page):
    head, body = page.split("</style>\n", 1)  # the template's one <style> block ends its head part
    assert "<title>" in head and "<div class=\"page\">" in body
    return SKELETON_HEAD + head + "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--standalone", metavar="PATH", help="also write a full HTML document there, e.g. a GitHub Pages index.html")
    main(ap.parse_args().standalone)
