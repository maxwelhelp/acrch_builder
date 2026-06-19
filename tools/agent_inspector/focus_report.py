#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def load_json(path, default):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="reports/agent_inspector/focused_report.md")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    req = load_json(root / "inspector_request.json", {})
    graph = load_json(root / "reports/agent_inspector/static_graph.json", {"nodes": [], "edges": []})
    roles = set(req.get("focus_roles", []))
    keywords = [x.lower() for x in req.get("focus_keywords", [])]
    files = req.get("focus_files", [])
    nodes = []
    for n in graph["nodes"]:
        blob = json.dumps(n, ensure_ascii=False).lower()
        if n.get("role") in roles or any(k in blob for k in keywords) or any(str(n.get("file", n.get("label", ""))).startswith(f) for f in files):
            nodes.append(n)
    ids = {n["id"] for n in nodes}
    edges = [e for e in graph["edges"] if e["source"] in ids or e["target"] in ids]
    lines = ["# Focused Report\n\n", "## Selected nodes\n\n"]
    for n in nodes[:100]:
        lines.append(f"- `{n.get('label')}` kind=`{n.get('kind')}` role=`{n.get('role')}` file=`{n.get('file','')}` line=`{n.get('line','')}`\n")
    lines.append("\n## Edges touching selected\n\n")
    for e in edges[:160]:
        lines.append(f"- `{e.get('source')}` --{e.get('kind')}--> `{e.get('target')}` label=`{e.get('label','')}` call=`{e.get('call','')}`\n")
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    print(f"[focus] wrote {out}")
if __name__ == "__main__":
    main()
