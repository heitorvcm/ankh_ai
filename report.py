"""Regenera relatorio.md (e curvas.png) de uma run:   python report.py --dir ~/ankh_runs/run1"""
import argparse
import os

from ankh_report import render_markdown

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    d = os.path.expanduser(a.dir)
    text = render_markdown(d, plot=not a.no_plot)
    print(text if text else f"nenhum report.jsonl em {d}")
