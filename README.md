# Twin Dilemma Explorer

Language models play a one-shot prisoner's dilemma against an exact copy of themselves. Nine models, four conditions, 25 samples per cell.

The interactive page: https://adb991.github.io/twin-dilemma/

Andrea Di Biagio set the question; Claude (Opus 5.5) built and ran the experiment, labelled the reasoning and built the page.

## Files

- `run.py`: the prompts and the two backends, Ollama for the local models and `claude -p` for the Claude models
- `classify.py`: keyword labels for the reasoning (correlation, dominance, other) and the summary tables
- `build_explorable.py`, `explorable_template.html`: build the page; `--standalone index.html` writes the version in this repo
- `results.jsonl`: the 900 answers behind the page, 9 models × 4 conditions × 25
- `results_claude_leaky_cwd.jsonl`: the first Claude run, from a temp folder named `twin-pd-…` that the models could see (the page's eval awareness section)
- `classified.jsonl`, `tables.txt`: the answers with their labels, and the tables
- `spotcheck.jsonl`, `spot_sample*.txt`: 88 hand labels with notes, and the answers read for them
- `run_*.log`: the run logs

## Rerun

```bash
python3 run.py --models qwen3.8:27b-mlx gemma3:27b llama3.2:3b gemma3:4b phi4-mini --n 25
python3 run.py --models claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5 claude-fable-5-1 --n 25 --workers 3
python3 classify.py
python3 build_explorable.py --standalone index.html
```

`run.py` skips samples already in `results.jsonl`; delete or move that file to start fresh. The local models need [Ollama](https://ollama.com) with the models pulled; the Claude models need a logged-in `claude` CLI.
