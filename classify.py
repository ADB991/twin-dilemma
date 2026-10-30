"""Classify the reasoning in results.jsonl and print the tables.

Two keyword flags per response:
  dom   the dominance argument: "regardless of what...", "no matter what",
        "dominant strategy", "in either case", ...
  corr  the correlation argument: the copy makes the same choice as me
        ("same reasoning", "mirror", "symmetric", "whatever I choose, it...")

The label is the argument that supports the final choice:
  COOPERATE + corr -> correlation     DEFECT + dom -> dominance     else -> other
so a defector that weighs the correlation argument and rejects it is still
"dominance". Writes classified.jsonl; --spot N prints a stratified random sample
for checking by hand.
"""
import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDITIONS = ["twin", "control", "sampling", "dominance"]

AGENT = r"(the other( player| ai| system)?|my (copy|twin|opponent|counterpart|clone)|the (copy|twin|opponent|clone)|they|it|he|she)"

DOM = [
    r"regardless of (what|whether|which|the (other|opponent|copy|twin)|my (copy|opponent|twin))",
    r"no matter (what|which|how)",
    rf"whatever {AGENT} (choose|chooses|does|do|pick|picks|decide|decides)",
    r"\bdominan(t|ce)\b", r"\bdominat(es|ed|ing)\b",
    r"in (either|both|every|any) (case|scenario)",
    r"always (better|higher|more|yields|gives|pays)",
    r"\bindividually rational\b",
]

CORR = [
    r"\bsame (reasoning|logic|decision|choice|conclusion|thought|process|answer|output|strategy|move|action|result|way)",
    r"\b(will|would|must|should) (also |likely |almost certainly |necessarily |inevitably )?(choose|make|reach|arrive at|come to|pick|select|output|produce|follow) the same",
    r"\b(mirror|mirrors|mirrored|mirroring)\b",
    r"\bsymmetr",
    r"\bcorrelat",
    r"\bsuperrational",
    r"\bconverge",
    r"(choices|decisions|outputs|answers|actions|moves) (will |would |must |should )?(likely |almost certainly |necessarily )?(match|be (the same|identical|aligned|correlated)|align|coincide)",
    r"whatever i (choose|decide|pick|do|reason)",
    rf"if i (choose to |decide to |pick |reason (that|to) )?(cooperate|defect)[^.]{{0,60}}?{AGENT} (will|would|should) (also |likely |too )?",
    r"only (two|2) (realistic |possible |plausible |likely |consistent )?(outcomes|options|possibilities|scenarios)",
    r"\b(identical|same) (inputs?|prompt)[^.]{0,80}(identical|same) (output|answer|choice|decision|result)",
    r"thinks? (exactly )?(like|as) (i|me) do",
    r"(both|each) of us (will )?(choose|reach|arrive)",
    r"\bfunctional decision|\bevidential decision|\bfdt\b|\bedt\b|\btimeless decision",
]

DOM_RE = [re.compile(p, re.I) for p in DOM]
CORR_RE = [re.compile(p, re.I) for p in CORR]


def reasoning(text):
    """Drop the FINAL ANSWER line so it cannot trip the patterns."""
    return re.sub(r"FINAL\s+ANSWER.*", "", text, flags=re.I | re.S)


def classify(rec):
    body = reasoning(rec["response"])
    dom = any(r.search(body) for r in DOM_RE)
    corr = any(r.search(body) for r in CORR_RE)
    choice = rec["choice"]
    if choice == "COOPERATE" and corr:
        label = "correlation"
    elif choice == "DEFECT" and dom:
        label = "dominance"
    else:
        label = "other"
    return {**rec, "dom": dom, "corr": corr, "label": label}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def tables(rows):
    models = list(dict.fromkeys(r["model"] for r in rows))
    cell = defaultdict(list)
    for r in rows:
        cell[(r["model"], r["condition"])].append(r)

    print("## cooperation rate: k/n (%) [95% Wilson]\n")
    print("| model | " + " | ".join(CONDITIONS) + " |")
    print("|---|" + "---|" * len(CONDITIONS))
    for m in models:
        out = []
        for c in CONDITIONS:
            rs = [r for r in cell[(m, c)] if r["choice"]]
            k, n = sum(r["choice"] == "COOPERATE" for r in rs), len(rs)
            lo, hi = wilson(k, n)
            unparsed = len(cell[(m, c)]) - n
            extra = f", {unparsed} unparsed" if unparsed else ""
            out.append(f"{k}/{n} ({100*k/n:.0f}%) [{100*lo:.0f}–{100*hi:.0f}]{extra}" if n else "–")
        print(f"| {m} | " + " | ".join(out) + " |")

    print("\n## label counts: correlation / dominance / other\n")
    print("| model | " + " | ".join(CONDITIONS) + " |")
    print("|---|" + "---|" * len(CONDITIONS))
    for m in models:
        out = []
        for c in CONDITIONS:
            cnt = Counter(r["label"] for r in cell[(m, c)])
            out.append(f"{cnt['correlation']} / {cnt['dominance']} / {cnt['other']}")
        print(f"| {m} | " + " | ".join(out) + " |")

    print("\n## flags among defectors and cooperators (all conditions)\n")
    print("| model | C: corr | C: neither | D: dom | D: dom+corr | D: neither |")
    print("|---|---|---|---|---|---|")
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        C = [r for r in rs if r["choice"] == "COOPERATE"]
        D = [r for r in rs if r["choice"] == "DEFECT"]
        print(f"| {m} | {sum(r['corr'] for r in C)}/{len(C)} | {sum(not r['corr'] for r in C)}/{len(C)} "
              f"| {sum(r['dom'] for r in D)}/{len(D)} | {sum(r['dom'] and r['corr'] for r in D)}/{len(D)} "
              f"| {sum(not r['dom'] for r in D)}/{len(D)} |")


def spot(rows, n, seed=0):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["label"])].append(r)
    picked = []
    keys = sorted(groups)
    while len(picked) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop(rng.randrange(len(groups[k]))))
    for r in picked:
        print(f"### {r['model']} | {r['condition']} | #{r['sample']} | choice={r['choice']} | "
              f"auto={r['label']} (dom={r['dom']}, corr={r['corr']})")
        print(reasoning(r["response"]).strip())
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot", type=int, default=0)
    ap.add_argument("--model", help="restrict the spot-check draw to one model")
    args = ap.parse_args()
    with (HERE / "results.jsonl").open() as f:
        rows = [classify(json.loads(l)) for l in f]
    with (HERE / "classified.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if args.spot:
        spot([r for r in rows if not args.model or r["model"] == args.model], args.spot)
    else:
        tables(rows)


if __name__ == "__main__":
    main()
