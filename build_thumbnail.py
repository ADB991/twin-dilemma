"""Build thumbnail.png: the results grid as a 1200x630 social preview image.

Draws the grid as a small HTML page from the same rows as the explorable, then
screenshots it with headless Chrome (a throwaway profile, so the user's own
Chrome is untouched).

Usage
  python3 build_thumbnail.py [OUT_DIR]
"""
import html
import subprocess
import sys
import tempfile
from pathlib import Path

from build_explorable import CONDITIONS, MODELS, load_rows

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]  # the page's light-mode ramp

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
* {{ box-sizing: border-box; margin: 0; }}
html, body {{ width: 1200px; height: 630px; overflow: hidden; }}
body {{ background: #f5f6f8; color: #10131a; padding: 40px 56px; display: grid; gap: 18px; align-content: start;
       font-family: "Bricolage Grotesque", "Avenir Next", system-ui, sans-serif; }}
.eyebrow {{ font-family: "IBM Plex Mono", Menlo, monospace; font-size: 14px; letter-spacing: 0.08em; text-transform: uppercase; color: #737a89; }}
h1 {{ font-size: 44px; font-weight: 700; letter-spacing: -0.015em; line-height: 1.05; }}
.sub {{ font-size: 23px; font-weight: 500; color: #4a5161; margin-top: 6px; }}
table {{ border-collapse: separate; border-spacing: 4px; width: 100%; margin-left: -4px; table-layout: fixed; }}
th {{ font-family: "IBM Plex Mono", Menlo, monospace; font-size: 14px; font-weight: 400; color: #4a5161; text-align: left; padding: 0 6px 2px; }}
tbody th {{ font-family: "Bricolage Grotesque", system-ui, sans-serif; font-size: 18px; font-weight: 500; color: #10131a; white-space: nowrap; width: 230px; }}
td {{ height: 36px; border-radius: 6px; text-align: center; font-variant-numeric: tabular-nums; padding: 0 8px; }}
td b {{ font-size: 19px; font-weight: 700; }}
td span {{ font-family: "IBM Plex Mono", Menlo, monospace; font-size: 12px; opacity: 0.85; margin-left: 8px; }}
</style></head><body>
<div>
  <p class="eyebrow">One-shot prisoner's dilemma · {n_models} models · {n_answers} answers · % that cooperate</p>
  <h1>LLM agents in the twin prisoner's dilemma</h1>
  <p class="sub">Do language models acausally coordinate with a copy of themselves?</p>
</div>
<table>
  <thead><tr><th style="width:230px"></th>{heads}</tr></thead>
  <tbody>{body}</tbody>
</table>
</body></html>
"""


def page():
    rows = load_rows()
    models = [m for m in MODELS if any(r["m"] == m["id"] for r in rows)]
    heads = "".join(f"<th>{html.escape(c['name'])}</th>" for c in CONDITIONS)
    body = ""
    for m in models:
        body += f"<tr><th>{html.escape(m['name'])}</th>"
        for c in CONDITIONS:
            rs = [r for r in rows if r["m"] == m["id"] and r["c"] == c["id"] and r["ch"]]
            k, n = sum(r["ch"] == "COOPERATE" for r in rs), len(rs)
            b = round(10 * k / n) if n else 0
            ink = "#ffffff" if b >= 7 else "#10131a"
            body += f'<td style="background:{SEQ[b]};color:{ink}"><b>{round(100 * k / n)}%</b><span>{k}/{n}</span></td>'
        body += "</tr>"
    return PAGE.format(n_models=len(models), n_answers=len(rows), heads=heads, body=body)


def main(out_dir):
    out = Path(out_dir).resolve() / "thumbnail.png"
    out.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "thumbnail.html"
        src.write_text(page())
        try:
            subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                            "--force-device-scale-factor=1", "--window-size=1200,630",
                            "--virtual-time-budget=8000", f"--user-data-dir={tmp}/profile",
                            f"--screenshot={out}", src.as_uri()],
                           capture_output=True, timeout=45)
        except subprocess.TimeoutExpired:
            pass  # on this machine Chrome writes the screenshot, then never exits; run() kills it
    if not out.exists() or out.stat().st_size == 0:
        raise SystemExit("no screenshot written")
    print(f"thumbnail: {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE)
