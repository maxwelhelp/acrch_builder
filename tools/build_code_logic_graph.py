#!/usr/bin/env python3
"""Build a code logic graph for agents and humans.

Outputs:
  reports/code_logic_graph.json  - machine-readable project graph
  reports/code_logic_graph.html  - interactive browser graph
  reports/CODE_MAP.md           - compact agent-readable architecture map

The goal is not runtime tracing. This is static code-logic mapping:
files -> classes -> functions -> calls -> state variables -> config/env/files.
Uses only Python standard library.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
    "runs", "checkpoints", "weights", "data", "datasets", "wandb", "node_modules",
}

ROLE_KEYWORDS = {
    "primitive": ["primitive", "topology", "embedding", "descriptor"],
    "scanner": ["scanner", "proposal", "candidate", "topk", "top_k", "semantic", "usage"],
    "simulator": ["simulator", "simulate", "sim", "predicted_gain", "gain"],
    "controller": ["controller", "choice", "logit", "gumbel", "select"],
    "executor": ["executor", "execute", "action", "matrix", "edge", "write_gate"],
    "credit": ["credit", "ablation", "ema", "real_gain", "staleness"],
    "memory": ["memory", "mem", "read", "write", "recall", "forget"],
    "report": ["report", "metrics", "trace", "json", "csv", "html", "writer"],
    "command": ["argparse", "main", "cli", "command", "run_"],
    "data": ["dataset", "loader", "batch", "train", "val"],
    "test": ["test", "validate", "smoke"],
}

IMPORTANT_EXTS = {".py", ".sh", ".md", ".toml", ".yaml", ".yml", ".json"}
PY_EXT = ".py"


@dataclass
class FunctionInfo:
    name: str
    qualname: str
    file: str
    lineno: int
    end_lineno: int | None
    class_name: str | None = None
    calls: list[str] = field(default_factory=list)
    self_reads: list[str] = field(default_factory=list)
    self_writes: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    env_reads: list[str] = field(default_factory=list)
    file_reads: list[str] = field(default_factory=list)
    file_writes: list[str] = field(default_factory=list)
    argparse_args: list[str] = field(default_factory=list)
    role: str = "unknown"


@dataclass
class FileInfo:
    path: str
    imports: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    role: str = "unknown"
    lines: int = 0


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & SKIP_DIRS)


def iter_project_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file() or should_skip(p):
            continue
        if p.suffix in IMPORTANT_EXTS:
            yield p


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def infer_role(*texts: str) -> str:
    joined = " ".join(texts).lower()
    scores = Counter()
    for role, words in ROLE_KEYWORDS.items():
        for w in words:
            if w in joined:
                scores[role] += 1
    if not scores:
        return "unknown"
    return scores.most_common(1)[0][0]


def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        return call_name(node.value)
    if isinstance(node, ast.Call):
        return call_name(node.func)
    return None


def literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def target_self_attr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, info: FunctionInfo):
        self.info = info

    def visit_Call(self, node: ast.Call) -> Any:
        name = call_name(node.func)
        if name:
            self.info.calls.append(name)

        # argparse: parser.add_argument("--foo")
        if name and name.endswith("add_argument"):
            for arg in node.args:
                s = literal_str(arg)
                if s and s.startswith("--"):
                    self.info.argparse_args.append(s)

        # environment reads: os.environ.get("X"), os.getenv("X")
        if name in {"os.getenv", "os.environ.get", "environ.get"}:
            if node.args:
                s = literal_str(node.args[0])
                if s:
                    self.info.env_reads.append(s)

        # open(path, mode)
        if name == "open" or name.endswith(".open"):
            mode = "r"
            if len(node.args) >= 2:
                mode = literal_str(node.args[1]) or mode
            for kw in node.keywords:
                if kw.arg == "mode":
                    mode = literal_str(kw.value) or mode
            path_s = literal_str(node.args[0]) if node.args else None
            marker = path_s or "<dynamic_path>"
            if any(ch in mode for ch in ["w", "a", "+"]):
                self.info.file_writes.append(marker)
            else:
                self.info.file_reads.append(marker)

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        attr = target_self_attr(node)
        if attr:
            self.info.self_reads.append(attr)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        for t in node.targets:
            self._visit_target(t)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        self._visit_target(node.target)
        if node.value:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        self._visit_target(node.target)
        self.visit(node.value)

    def _visit_target(self, node: ast.AST) -> None:
        attr = target_self_attr(node)
        if attr:
            self.info.self_writes.append(attr)
        for child in ast.iter_child_nodes(node):
            self._visit_target(child)


def parse_python_file(path: Path, root: Path) -> tuple[FileInfo, list[FunctionInfo]]:
    text = read_text_safe(path)
    rel = relpath(path, root)
    finfo = FileInfo(path=rel, lines=len(text.splitlines()))
    functions: list[FunctionInfo] = []
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError as e:
        finfo.role = "syntax_error"
        return finfo, functions

    parent_class: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                finfo.imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                finfo.imports.append(node.module)
        elif isinstance(node, ast.ClassDef):
            finfo.classes.append(node.name)

    class Parser(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            parent_class.append(node.name)
            self.generic_visit(node)
            parent_class.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            cls = parent_class[-1] if parent_class else None
            qn = f"{cls}.{node.name}" if cls else node.name
            args = [a.arg for a in node.args.args]
            info = FunctionInfo(
                name=node.name,
                qualname=qn,
                file=rel,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", None),
                class_name=cls,
                args=args,
            )
            FunctionVisitor(info).visit(node)
            info.calls = sorted(set(info.calls))
            info.self_reads = sorted(set(info.self_reads))
            info.self_writes = sorted(set(info.self_writes))
            info.env_reads = sorted(set(info.env_reads))
            info.file_reads = sorted(set(info.file_reads))
            info.file_writes = sorted(set(info.file_writes))
            info.argparse_args = sorted(set(info.argparse_args))
            info.role = infer_role(rel, qn, " ".join(info.calls), " ".join(info.self_writes))
            functions.append(info)
            finfo.functions.append(qn)
            # Do not recurse into nested defs as separate top-level nodes for now.

        visit_AsyncFunctionDef = visit_FunctionDef

    Parser().visit(tree)
    finfo.imports = sorted(set(finfo.imports))
    finfo.classes = sorted(set(finfo.classes))
    finfo.role = infer_role(rel, " ".join(finfo.classes), " ".join(finfo.functions), " ".join(finfo.imports))
    return finfo, functions


def analyze_repo(root: Path) -> dict[str, Any]:
    files: dict[str, FileInfo] = {}
    funcs: list[FunctionInfo] = []
    non_py_files: list[FileInfo] = []

    for path in iter_project_files(root):
        rel = relpath(path, root)
        if path.suffix == PY_EXT:
            f, fs = parse_python_file(path, root)
            files[rel] = f
            funcs.extend(fs)
        else:
            text = read_text_safe(path)
            f = FileInfo(path=rel, lines=len(text.splitlines()), role=infer_role(rel, text[:2000]))
            files[rel] = f
            non_py_files.append(f)

    return build_graph(files, funcs)


def build_graph(files: dict[str, FileInfo], funcs: list[FunctionInfo]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_node(node_id: str, label: str, kind: str, **attrs: Any) -> None:
        nodes[node_id] = {"id": node_id, "label": label, "kind": kind, **attrs}

    def add_edge(src: str, dst: str, kind: str, **attrs: Any) -> None:
        edges.append({"source": src, "target": dst, "kind": kind, **attrs})

    for path, f in sorted(files.items()):
        fid = f"file:{path}"
        add_node(fid, path, "file", role=f.role, lines=f.lines)
        for imp in f.imports[:80]:
            iid = f"import:{imp}"
            add_node(iid, imp, "import")
            add_edge(fid, iid, "imports")
        for cls in f.classes:
            cid = f"class:{path}:{cls}"
            add_node(cid, cls, "class", file=path, role=infer_role(path, cls))
            add_edge(fid, cid, "defines")

    func_by_simple: dict[str, list[str]] = defaultdict(list)
    func_by_qual: dict[str, str] = {}
    for fn in funcs:
        fid = f"func:{fn.file}:{fn.qualname}"
        func_by_simple[fn.name].append(fid)
        func_by_qual[fn.qualname] = fid

    for fn in funcs:
        fid = f"func:{fn.file}:{fn.qualname}"
        add_node(fid, fn.qualname, "function", file=fn.file, role=fn.role, line=fn.lineno)
        add_edge(f"file:{fn.file}", fid, "defines")
        if fn.class_name:
            add_edge(f"class:{fn.file}:{fn.class_name}", fid, "owns")

        for call in fn.calls:
            short = call.split(".")[-1]
            candidates = func_by_simple.get(short, [])
            if candidates:
                for dst in candidates[:4]:
                    if dst != fid:
                        add_edge(fid, dst, "calls", call=call)
            else:
                cid = f"external_call:{call}"
                add_node(cid, call, "external_call")
                add_edge(fid, cid, "calls_external")

        for attr in fn.self_reads:
            sid = f"state:{fn.file}:{fn.class_name or '<module>'}:{attr}"
            add_node(sid, attr, "state", file=fn.file, owner=fn.class_name)
            add_edge(fid, sid, "reads_state")
        for attr in fn.self_writes:
            sid = f"state:{fn.file}:{fn.class_name or '<module>'}:{attr}"
            add_node(sid, attr, "state", file=fn.file, owner=fn.class_name)
            add_edge(fid, sid, "writes_state")
        for arg in fn.argparse_args:
            aid = f"arg:{arg}"
            add_node(aid, arg, "config_arg")
            add_edge(fid, aid, "defines_arg")
        for env in fn.env_reads:
            eid = f"env:{env}"
            add_node(eid, env, "env")
            add_edge(fid, eid, "reads_env")
        for p in fn.file_reads:
            rid = f"io_read:{p}"
            add_node(rid, p, "io")
            add_edge(fid, rid, "reads_file")
        for p in fn.file_writes:
            wid = f"io_write:{p}"
            add_node(wid, p, "io")
            add_edge(fid, wid, "writes_file")

    role_counts = Counter(n.get("role", "unknown") for n in nodes.values() if n["kind"] in {"file", "function", "class"})
    kind_counts = Counter(n["kind"] for n in nodes.values())
    edge_counts = Counter(e["kind"] for e in edges)

    graph = {
        "schema": "acrch_builder.code_logic_graph.v1",
        "summary": {
            "nodes": len(nodes),
            "edges": len(edges),
            "files": sum(1 for n in nodes.values() if n["kind"] == "file"),
            "functions": sum(1 for n in nodes.values() if n["kind"] == "function"),
            "classes": sum(1 for n in nodes.values() if n["kind"] == "class"),
            "role_counts": dict(role_counts),
            "kind_counts": dict(kind_counts),
            "edge_counts": dict(edge_counts),
        },
        "nodes": list(nodes.values()),
        "edges": edges,
    }
    return graph


def write_json(graph: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")


def top_items(counter: Counter, n: int = 20) -> str:
    if not counter:
        return "- none\n"
    return "".join(f"- `{k}`: {v}\n" for k, v in counter.most_common(n))


def write_markdown(graph: dict[str, Any], path: Path) -> None:
    nodes = graph["nodes"]
    edges = graph["edges"]
    file_nodes = [n for n in nodes if n["kind"] == "file"]
    func_nodes = [n for n in nodes if n["kind"] == "function"]
    role_counts = Counter(n.get("role", "unknown") for n in file_nodes + func_nodes)
    edge_counts = Counter(e["kind"] for e in edges)

    incoming = Counter(e["target"] for e in edges)
    outgoing = Counter(e["source"] for e in edges)
    important_funcs = sorted(
        [n for n in func_nodes],
        key=lambda n: incoming[n["id"]] + outgoing[n["id"]],
        reverse=True,
    )[:30]

    files_by_role: dict[str, list[str]] = defaultdict(list)
    for n in file_nodes:
        files_by_role[n.get("role", "unknown")].append(n["label"])

    lines = []
    lines.append("# CODE_MAP\n\n")
    lines.append("Generated by `tools/build_code_logic_graph.py`.\n\n")
    lines.append("## Summary\n\n")
    s = graph["summary"]
    lines.append(f"- nodes: `{s['nodes']}`\n")
    lines.append(f"- edges: `{s['edges']}`\n")
    lines.append(f"- files: `{s['files']}`\n")
    lines.append(f"- classes: `{s['classes']}`\n")
    lines.append(f"- functions: `{s['functions']}`\n\n")

    lines.append("## Roles detected\n\n")
    lines.append(top_items(role_counts))
    lines.append("\n## Edge types\n\n")
    lines.append(top_items(edge_counts))

    lines.append("\n## Files by role\n\n")
    for role, paths in sorted(files_by_role.items()):
        lines.append(f"### {role}\n\n")
        for p in sorted(paths)[:80]:
            lines.append(f"- `{p}`\n")
        lines.append("\n")

    lines.append("## Most connected functions/classes\n\n")
    for n in important_funcs:
        degree = incoming[n["id"]] + outgoing[n["id"]]
        lines.append(f"- `{n['label']}` in `{n.get('file')}` role=`{n.get('role')}` degree=`{degree}` line=`{n.get('line')}`\n")

    lines.append("\n## How to use this map for agent edits\n\n")
    lines.append("Before changing code, find the target role and inspect the connected nodes in `reports/code_logic_graph.html`.\n")
    lines.append("For risky edits, check: callers, state writes, config args, report outputs, and validation commands.\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def write_html(graph: dict[str, Any], path: Path) -> None:
    data = json.dumps(graph, ensure_ascii=False)
    escaped = html.escape(data)
    template = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>Code Logic Graph</title>
<style>
body {{ margin:0; font-family: system-ui, sans-serif; background:#111; color:#eee; }}
#top {{ padding:12px 16px; border-bottom:1px solid #333; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
#graph {{ width:100vw; height:calc(100vh - 76px); }}
input, select, button {{ background:#1f1f1f; color:#eee; border:1px solid #444; padding:6px 8px; border-radius:6px; }}
.badge {{ color:#bbb; font-size:12px; }}
.node text {{ pointer-events:none; font-size:11px; fill:#eee; }}
.link {{ stroke:#777; stroke-opacity:.35; }}
.tooltip {{ position:fixed; pointer-events:none; background:#222; border:1px solid #555; padding:8px; border-radius:8px; max-width:520px; font-size:12px; display:none; }}
</style>
</head>
<body>
<div id=\"top\">
  <strong>Code Logic Graph</strong>
  <label>Role <select id=\"role\"><option value=\"\">all</option></select></label>
  <label>Kind <select id=\"kind\"><option value=\"\">all</option></select></label>
  <label>Search <input id=\"search\" placeholder=\"file/function/state\" /></label>
  <button id=\"reset\">reset</button>
  <span class=\"badge\" id=\"stats\"></span>
</div>
<svg id=\"graph\"></svg>
<div class=\"tooltip\" id=\"tip\"></div>
<script src=\"https://cdn.jsdelivr.net/npm/d3@7\"></script>
<script id=\"graph-data\" type=\"application/json\">{escaped}</script>
<script>
const graph = JSON.parse(document.getElementById('graph-data').textContent);
const svg = d3.select('#graph');
const width = window.innerWidth;
const height = window.innerHeight - 76;
const color = d3.scaleOrdinal()
  .domain(['file','class','function','state','config_arg','env','io','import','external_call'])
  .range(['#4e79a7','#f28e2b','#59a14f','#e15759','#b07aa1','#76b7b2','#edc949','#9c755f','#bab0ab']);
const roleSel = document.getElementById('role');
const kindSel = document.getElementById('kind');
const searchInp = document.getElementById('search');
const stats = document.getElementById('stats');
const tip = document.getElementById('tip');

for (const r of Array.from(new Set(graph.nodes.map(n => n.role).filter(Boolean))).sort()) roleSel.add(new Option(r, r));
for (const k of Array.from(new Set(graph.nodes.map(n => n.kind))).sort()) kindSel.add(new Option(k, k));

function filtered() {
  const role = roleSel.value;
  const kind = kindSel.value;
  const q = searchInp.value.toLowerCase();
  const keep = new Set(graph.nodes.filter(n =>
    (!role || n.role === role) && (!kind || n.kind === kind) && (!q || JSON.stringify(n).toLowerCase().includes(q))
  ).map(n => n.id));
  const links = graph.edges.filter(e => keep.has(e.source) && keep.has(e.target));
  const linked = new Set();
  links.forEach(e => { linked.add(e.source); linked.add(e.target); });
  const nodes = graph.nodes.filter(n => keep.has(n.id) && (linked.has(n.id) || keep.size < 300));
  return {nodes, links};
}

function render() {
  svg.selectAll('*').remove();
  const {nodes, links} = filtered();
  stats.textContent = `${nodes.length} nodes / ${links.length} edges`;
  const zoomLayer = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.1, 5]).on('zoom', e => zoomLayer.attr('transform', e.transform)));
  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(d => d.kind === 'defines' ? 55 : 95).strength(.35))
    .force('charge', d3.forceManyBody().strength(-180))
    .force('center', d3.forceCenter(width/2, height/2))
    .force('collision', d3.forceCollide().radius(d => d.kind === 'file' ? 22 : 14));
  const link = zoomLayer.append('g').selectAll('line').data(links).join('line').attr('class','link').attr('stroke-width', d => d.kind === 'calls' ? 1.4 : .8);
  const node = zoomLayer.append('g').selectAll('g').data(nodes).join('g').attr('class','node').call(drag(sim));
  node.append('circle').attr('r', d => d.kind === 'file' ? 10 : d.kind === 'function' ? 7 : 6).attr('fill', d => color(d.kind));
  node.append('text').text(d => d.label.length > 34 ? d.label.slice(0, 31) + '…' : d.label).attr('x', 10).attr('y', 4);
  node.on('mousemove', (event, d) => {
    tip.style.display = 'block';
    tip.style.left = (event.clientX + 12) + 'px';
    tip.style.top = (event.clientY + 12) + 'px';
    tip.innerHTML = `<b>${d.label}</b><br>kind: ${d.kind}<br>role: ${d.role || ''}<br>file: ${d.file || ''}<br>line: ${d.line || ''}<br><small>${d.id}</small>`;
  }).on('mouseleave', () => tip.style.display = 'none');
  sim.on('tick', () => {
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });
}
function drag(sim) {
  return d3.drag()
    .on('start', (event, d) => { if (!event.active) sim.alphaTarget(.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
    .on('end', (event, d) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; });
}
roleSel.onchange = render; kindSel.onchange = render; searchInp.oninput = render; document.getElementById('reset').onclick = () => { roleSel.value=''; kindSel.value=''; searchInp.value=''; render(); };
render();
</script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build static code logic graph for acrch_builder.")
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--out-json", default="reports/code_logic_graph.json")
    ap.add_argument("--out-html", default="reports/code_logic_graph.html")
    ap.add_argument("--out-md", default="reports/CODE_MAP.md")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    graph = analyze_repo(root)
    write_json(graph, root / args.out_json)
    write_html(graph, root / args.out_html)
    write_markdown(graph, root / args.out_md)
    print(f"[code-map] files={graph['summary']['files']} functions={graph['summary']['functions']} nodes={graph['summary']['nodes']} edges={graph['summary']['edges']}")
    print(f"[code-map] wrote {args.out_json}")
    print(f"[code-map] wrote {args.out_html}")
    print(f"[code-map] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
