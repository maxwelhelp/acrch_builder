#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", "dist", "build", ".tox", ".ruff_cache",
    "runs", "checkpoints", "weights", "datasets", "wandb",
}
DEFAULT_CODE_EXTS = {".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".json"}

DEFAULT_ROLES = {
    "entrypoint": {"keywords": ["main", "argparse", "cli", "command", "run_", "train", "validate"], "description": "Starts workflows and parses config."},
    "model_core": {"keywords": ["model", "layer", "forward", "backbone", "network", "module", "state", "slot"], "description": "Main computation path."},
    "controller": {"keywords": ["controller", "choice", "decision", "gate", "router", "select", "policy", "logits", "fanout", "boundary"], "description": "Chooses actions/routes/modes."},
    "memory": {"keywords": ["memory", "mem", "cache", "read", "write", "forget", "recall", "w_r", "w_w"], "description": "Persistent/recurrent state."},
    "credit": {"keywords": ["credit", "ablation", "delta_loss", "gain", "ema", "reward", "score", "collector"], "description": "Feedback and usefulness signal."},
    "council": {"keywords": ["council", "simulator", "simulate", "sim", "preview", "predicted_gain", "aux", "quality"], "description": "Predicts/calibrates consequences."},
    "scanner": {"keywords": ["scanner", "proposal", "candidate", "search", "topk", "top_k", "semantic", "usage"], "description": "Finds candidate actions/objects."},
    "executor": {"keywords": ["executor", "execute", "apply", "operation", "primitive", "action", "edge"], "description": "Runs selected operations."},
    "data": {"keywords": ["dataset", "dataloader", "batch", "sample", "loader", "input", "target", "synthetic"], "description": "Data pipeline."},
    "reporting": {"keywords": ["report", "metrics", "trace", "json", "csv", "html", "logger", "writer"], "description": "Writes diagnostics."},
}
DEFAULT_LOOPS = {
    "learning_loop": {
        "description": "decision -> computation -> feedback -> prediction/calibration -> decision",
        "roles": ["controller", "model_core", "credit", "council", "controller"],
        "required": False,
    }
}

@dataclass
class FileInfo:
    path: str
    ext: str
    role: str = "unknown"
    lines: int = 0
    imports: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)

@dataclass
class FuncInfo:
    label: str
    file: str
    line: int
    role: str = "unknown"
    class_name: str | None = None
    calls: list[str] = field(default_factory=list)
    self_reads: list[str] = field(default_factory=list)
    self_writes: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    file_reads: list[str] = field(default_factory=list)
    file_writes: list[str] = field(default_factory=list)

def load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except Exception:
        try:
            return json.loads(text)
        except Exception as e:
            raise RuntimeError(f"Cannot parse {path}. Install pyyaml or write JSON config.") from e

def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(root: Path) -> dict[str, Any]:
    cfg = {
        "project_name": root.name,
        "skip_dirs": sorted(DEFAULT_SKIP_DIRS),
        "code_extensions": sorted(DEFAULT_CODE_EXTS),
        "roles": DEFAULT_ROLES,
        "closed_loops": DEFAULT_LOOPS,
        "max_nodes_html": 900,
    }
    return deep_merge(cfg, load_config_file(root / ".codegraph.yml"))

def rel(p: Path, root: Path) -> str:
    return p.resolve().relative_to(root.resolve()).as_posix()

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def should_skip(p: Path, root: Path, cfg: dict[str, Any]) -> bool:
    try:
        rp = p.relative_to(root)
    except ValueError:
        rp = p
    return bool(set(rp.parts) & set(cfg.get("skip_dirs", [])))

def role_of(cfg: dict[str, Any], *parts: str) -> str:
    s = " ".join(x for x in parts if x).lower()
    scores = Counter()
    for role, spec in cfg.get("roles", {}).items():
        for kw in spec.get("keywords", []):
            if str(kw).lower() in s:
                scores[role] += 1
    return scores.most_common(1)[0][0] if scores else "unknown"

def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        p = call_name(node.value)
        return f"{p}.{node.attr}" if p else node.attr
    if isinstance(node, ast.Call):
        return call_name(node.func)
    if isinstance(node, ast.Subscript):
        return call_name(node.value)
    return None

def lit(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

def self_attr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None

class FuncVisitor(ast.NodeVisitor):
    def __init__(self, info: FuncInfo):
        self.info = info

    def visit_Call(self, node: ast.Call) -> Any:
        name = call_name(node.func)
        if name:
            self.info.calls.append(name)
        if name and name.endswith("add_argument"):
            for a in node.args:
                s = lit(a)
                if s and s.startswith("--"):
                    self.info.args.append(s)
        if name in {"os.getenv", "os.environ.get", "environ.get"} and node.args:
            s = lit(node.args[0])
            if s:
                self.info.env.append(s)
        if name == "open" or (name and name.endswith(".open")):
            mode = "r"
            if len(node.args) >= 2:
                mode = lit(node.args[1]) or mode
            p = lit(node.args[0]) if node.args else "<dynamic_path>"
            if any(x in mode for x in ["w", "a", "+"]):
                self.info.file_writes.append(p or "<dynamic_path>")
            else:
                self.info.file_reads.append(p or "<dynamic_path>")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        a = self_attr(node)
        if a:
            self.info.self_reads.append(a)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        for t in node.targets:
            self._target(t)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        self._target(node.target)
        if node.value:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        self._target(node.target)
        self.visit(node.value)

    def _target(self, node: ast.AST) -> None:
        a = self_attr(node)
        if a:
            self.info.self_writes.append(a)
        for c in ast.iter_child_nodes(node):
            self._target(c)

def parse_py(path: Path, root: Path, cfg: dict[str, Any]) -> tuple[FileInfo, list[FuncInfo]]:
    text = read_text(path)
    rp = rel(path, root)
    fi = FileInfo(path=rp, ext=path.suffix, lines=len(text.splitlines()))
    out: list[FuncInfo] = []
    try:
        tree = ast.parse(text, filename=rp)
    except SyntaxError:
        fi.role = "syntax_error"
        return fi, out
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            fi.imports.extend(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            fi.imports.append(n.module)
        elif isinstance(n, ast.ClassDef):
            fi.classes.append(n.name)
    cls_stack: list[str] = []
    class V(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            cls_stack.append(node.name)
            self.generic_visit(node)
            cls_stack.pop()
        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            cls = cls_stack[-1] if cls_stack else None
            label = f"{cls}.{node.name}" if cls else node.name
            info = FuncInfo(label=label, file=rp, line=node.lineno, class_name=cls)
            FuncVisitor(info).visit(node)
            for field_name in ["calls", "self_reads", "self_writes", "args", "env", "file_reads", "file_writes"]:
                setattr(info, field_name, sorted(set(getattr(info, field_name))))
            info.role = role_of(cfg, rp, label, " ".join(info.calls), " ".join(info.self_writes), " ".join(info.args))
            out.append(info)
            fi.functions.append(label)
        visit_AsyncFunctionDef = visit_FunctionDef
    V().visit(tree)
    fi.imports = sorted(set(fi.imports))
    fi.classes = sorted(set(fi.classes))
    fi.functions = sorted(set(fi.functions))
    fi.role = role_of(cfg, rp, " ".join(fi.classes), " ".join(fi.functions), " ".join(fi.imports))
    return fi, out

def project_files(root: Path, cfg: dict[str, Any]) -> list[Path]:
    exts = set(cfg.get("code_extensions", sorted(DEFAULT_CODE_EXTS)))
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in exts and not should_skip(p, root, cfg))

def build_graph(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    files: list[FileInfo] = []
    funcs: list[FuncInfo] = []

    def node(i: str, label: str, kind: str, **kw: Any) -> None:
        nodes[i] = {"id": i, "label": label, "kind": kind, **kw}

    def edge(s: str, t: str, kind: str, **kw: Any) -> None:
        if s in nodes and t in nodes:
            edges.append({"source": s, "target": t, "kind": kind, **kw})

    for p in project_files(root, cfg):
        rp = rel(p, root)
        if p.suffix == ".py":
            fi, fns = parse_py(p, root, cfg)
        else:
            text = read_text(p)
            fi, fns = FileInfo(path=rp, ext=p.suffix, role=role_of(cfg, rp, text[:3000]), lines=len(text.splitlines())), []
        files.append(fi)
        funcs.extend(fns)

    by_simple: dict[str, list[str]] = defaultdict(list)
    for fn in funcs:
        by_simple[fn.label.split(".")[-1]].append(f"func:{fn.file}:{fn.label}")

    for fi in files:
        fid = f"file:{fi.path}"
        node(fid, fi.path, "file", role=fi.role, lines=fi.lines, ext=fi.ext)
        for imp in fi.imports:
            iid = f"import:{imp}"
            node(iid, imp, "import", role="external")
            edge(fid, iid, "imports")
        for c in fi.classes:
            cid = f"class:{fi.path}:{c}"
            node(cid, c, "class", file=fi.path, role=role_of(cfg, fi.path, c))
            edge(fid, cid, "defines")

    for fn in funcs:
        fid = f"func:{fn.file}:{fn.label}"
        node(fid, fn.label, "function", file=fn.file, line=fn.line, role=fn.role, class_name=fn.class_name)
        edge(f"file:{fn.file}", fid, "defines")
        if fn.class_name:
            edge(f"class:{fn.file}:{fn.class_name}", fid, "owns")
        for call in fn.calls:
            short = call.split(".")[-1]
            targets = by_simple.get(short)
            if targets:
                for dst in targets[:4]:
                    if dst != fid:
                        edge(fid, dst, "calls", call=call)
            elif not call.startswith(("append", "extend", "split", "join", "items", "values", "keys", "get")):
                eid = f"external:{call}"
                node(eid, call, "external_call", role="external")
                edge(fid, eid, "calls_external", call=call)
        for a in fn.self_reads:
            sid = f"state:{fn.file}:{fn.class_name or 'module'}:{a}"
            node(sid, a, "state", file=fn.file, owner=fn.class_name, role=role_of(cfg, a))
            edge(fid, sid, "reads_state")
        for a in fn.self_writes:
            sid = f"state:{fn.file}:{fn.class_name or 'module'}:{a}"
            node(sid, a, "state", file=fn.file, owner=fn.class_name, role=role_of(cfg, a))
            edge(fid, sid, "writes_state")
        for a in fn.args:
            aid = f"arg:{a}"
            node(aid, a, "config_arg", role="entrypoint")
            edge(fid, aid, "defines_arg")
        for evar in fn.env:
            eid = f"env:{evar}"
            node(eid, evar, "env", role="entrypoint")
            edge(fid, eid, "reads_env")
        for fr in fn.file_reads:
            rid = f"io_read:{fr}"
            node(rid, fr, "io", role="reporting")
            edge(fid, rid, "reads_file")
        for fw in fn.file_writes:
            wid = f"io_write:{fw}"
            node(wid, fw, "io", role="reporting")
            edge(fid, wid, "writes_file")

    role_members: dict[str, list[str]] = defaultdict(list)
    for n in list(nodes.values()):
        r = n.get("role")
        if r and r not in {"unknown", "external"}:
            role_members[r].append(n["id"])

    for r in sorted(role_members):
        rid = f"role:{r}"
        desc = cfg.get("roles", {}).get(r, {}).get("description", "")
        node(rid, r, "role", role=r, description=desc)
        for m in role_members[r]:
            edge(rid, m, "role_contains")

    for loop_name, spec in cfg.get("closed_loops", {}).items():
        roles = spec.get("roles", [])
        for i in range(len(roles) - 1):
            a, b = roles[i], roles[i + 1]
            if f"role:{a}" in nodes and f"role:{b}" in nodes:
                edges.append({"source": f"role:{a}", "target": f"role:{b}", "kind": "expected_loop", "loop": loop_name, "label": f"{a} -> {b}"})

    graph = {
        "schema": "agent_codegraph.v1",
        "project": cfg.get("project_name", root.name),
        "config_used": {"roles": cfg.get("roles", {}), "closed_loops": cfg.get("closed_loops", {})},
        "nodes": list(nodes.values()),
        "edges": edges,
    }
    graph["summary"] = summarize(graph)
    graph["analysis"] = analyze_graph(graph, cfg)
    return graph

def summarize(graph: dict[str, Any]) -> dict[str, Any]:
    nodes, edges = graph["nodes"], graph["edges"]
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "files": sum(1 for n in nodes if n["kind"] == "file"),
        "functions": sum(1 for n in nodes if n["kind"] == "function"),
        "classes": sum(1 for n in nodes if n["kind"] == "class"),
        "roles": dict(Counter(n.get("role", "unknown") for n in nodes)),
        "kinds": dict(Counter(n["kind"] for n in nodes)),
        "edge_kinds": dict(Counter(e["kind"] for e in edges)),
    }

def analyze_graph(graph: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    inc = Counter(e["target"] for e in edges)
    out = Counter(e["source"] for e in edges)
    degree = {nid: inc[nid] + out[nid] for nid in nodes}
    risk_nodes = sorted(nodes.values(), key=lambda n: degree.get(n["id"], 0), reverse=True)[:50]
    loop_reports = {}
    for loop_name, spec in cfg.get("closed_loops", {}).items():
        roles = spec.get("roles", [])
        missing_roles = [r for r in roles if f"role:{r}" not in nodes]
        missing_edges = []
        for i in range(len(roles) - 1):
            a, b = roles[i], roles[i + 1]
            found = any(e["kind"] == "expected_loop" and e["source"] == f"role:{a}" and e["target"] == f"role:{b}" for e in edges)
            if not found:
                missing_edges.append(f"{a}->{b}")
        loop_reports[loop_name] = {
            "description": spec.get("description", ""),
            "roles": roles,
            "missing_roles": missing_roles,
            "missing_expected_edges": missing_edges,
            "closed": not missing_roles and not missing_edges and bool(roles),
        }
    holes = []
    role_counts = Counter(n.get("role", "unknown") for n in nodes.values())
    for role in cfg.get("roles", {}):
        if role_counts.get(role, 0) == 0:
            holes.append({"severity": "warning", "kind": "missing_role", "message": f"No nodes detected for role `{role}`."})
    for n in nodes.values():
        if n["kind"] in {"function", "class"} and degree.get(n["id"], 0) <= 1 and n.get("role") not in {"unknown", "external"}:
            holes.append({"severity": "info", "kind": "weakly_connected_role_node", "message": f"`{n['label']}` role `{n.get('role')}` is weakly connected.", "node": n["id"]})
    role_edges = Counter()
    for e in edges:
        s, t = nodes.get(e["source"]), nodes.get(e["target"])
        if not s or not t:
            continue
        sr, tr = s.get("role", "unknown"), t.get("role", "unknown")
        if sr != tr and sr not in {"unknown", "external"} and tr not in {"unknown", "external"}:
            role_edges[(sr, tr)] += 1
    return {
        "loop_reports": loop_reports,
        "holes": holes[:200],
        "risk_nodes": [{"id": n["id"], "label": n["label"], "kind": n["kind"], "role": n.get("role", ""), "degree": degree.get(n["id"], 0), "file": n.get("file", ""), "line": n.get("line", "")} for n in risk_nodes],
        "role_coupling": [{"source_role": a, "target_role": b, "count": c} for (a, b), c in role_edges.most_common(100)],
    }

def write_csv_edges(graph: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "target", "kind", "label", "loop", "call"])
        w.writeheader()
        for e in graph["edges"]:
            w.writerow({k: e.get(k, "") for k in ["source", "target", "kind", "label", "loop", "call"]})

def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    out = ["|" + "|".join(cols) + "|\n", "|" + "|".join(["---"] * len(cols)) + "|\n"]
    for r in rows:
        out.append("|" + "|".join(str(r.get(c, "")).replace("|", "\\|") for c in cols) + "|\n")
    return "".join(out)

def write_agent_brief(graph: dict[str, Any], path: Path) -> None:
    s, a = graph["summary"], graph["analysis"]
    lines = [f"# Agent Code Brief: {graph.get('project')}\n\n", "First file for an agent before editing.\n\n"]
    lines.append("## Static summary\n\n")
    lines.append(f"- nodes: `{s['nodes']}`\n- edges: `{s['edges']}`\n- files: `{s['files']}`\n- functions: `{s['functions']}`\n- classes: `{s['classes']}`\n\n")
    lines.append("## Closed-loop status\n\n")
    for name, rep in a["loop_reports"].items():
        status = "CLOSED" if rep["closed"] else "BROKEN/INCOMPLETE"
        lines.append(f"### `{name}`: **{status}**\n\n")
        if rep.get("description"):
            lines.append(rep["description"] + "\n\n")
        lines.append(f"- roles: `{' -> '.join(rep['roles'])}`\n")
        lines.append(f"- missing roles: `{rep['missing_roles']}`\n")
        lines.append(f"- missing expected edges: `{rep['missing_expected_edges']}`\n\n")
    lines.append("## Main detected roles\n\n")
    rows = [{"role": k, "count": v} for k, v in sorted(s["roles"].items(), key=lambda kv: (-kv[1], kv[0])) if k not in {"unknown", "external"}]
    lines.append(md_table(rows[:30], ["role", "count"]) + "\n")
    lines.append("## Highest edit-risk nodes\n\n")
    lines.append(md_table(a["risk_nodes"][:25], ["label", "kind", "role", "degree", "file", "line"]) + "\n")
    lines.append("## Possible holes / warnings\n\n")
    if a["holes"]:
        for h in a["holes"][:60]:
            lines.append(f"- `{h['severity']}` `{h['kind']}`: {h['message']}\n")
    else:
        lines.append("- none detected\n")
    lines.append("\n## How to go deeper\n\n")
    lines.append("Open `reports/code_graph_dashboard.html` and switch views: `agent_brief`, `loop`, `architecture`, `calls`, `state`, `io`, `imports`, `risk`, `full`.\n")
    path.write_text("".join(lines), encoding="utf-8")

def write_hole_report(graph: dict[str, Any], path: Path) -> None:
    a = graph["analysis"]
    lines = ["# Code Graph Holes Report\n\n"]
    for name, rep in a["loop_reports"].items():
        lines.append(f"## Loop `{name}`\n\n- closed: `{rep['closed']}`\n- missing roles: `{rep['missing_roles']}`\n- missing expected edges: `{rep['missing_expected_edges']}`\n\n")
    lines.append("## Warnings\n\n")
    if not a["holes"]:
        lines.append("- none\n")
    for h in a["holes"]:
        lines.append(f"- `{h['severity']}` `{h['kind']}`: {h['message']}\n")
    path.write_text("".join(lines), encoding="utf-8")

HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Agent Code Graph</title>
<style>
body{margin:0;background:#111217;color:#eee;font-family:system-ui,sans-serif}
#top{padding:10px 14px;border-bottom:1px solid #333;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
#canvas{width:100vw;height:calc(100vh - 70px);display:block}
select,input,button{background:#1f2028;color:#eee;border:1px solid #444;border-radius:6px;padding:6px}
.tip{position:fixed;display:none;pointer-events:none;background:#222;border:1px solid #555;border-radius:8px;padding:8px;max-width:560px;font-size:12px;z-index:9}
.small{font-size:12px;color:#aaa}
line{stroke:#777;stroke-opacity:.38}
text{fill:#eee;font-size:11px;pointer-events:none}
</style></head>
<body>
<div id="top"><b>Agent Code Graph</b>
<label>view <select id="view"><option>agent_brief</option><option>loop</option><option>architecture</option><option>calls</option><option>state</option><option>io</option><option>imports</option><option>risk</option><option>full</option></select></label>
<label>role <select id="role"><option value="">all</option></select></label>
<label>kind <select id="kind"><option value="">all</option></select></label>
<input id="q" placeholder="search"><button id="reset">reset</button><span id="stats" class="small"></span></div>
<svg id="canvas"></svg><div id="tip" class="tip"></div>
<script id="graph-data" type="application/json">__GRAPH_JSON__</script>
<script>
const graph = JSON.parse(document.getElementById('graph-data').textContent);
const svg = document.getElementById('canvas'), viewSel = document.getElementById('view'), roleSel = document.getElementById('role'), kindSel = document.getElementById('kind'), qInp = document.getElementById('q'), stats = document.getElementById('stats'), tip = document.getElementById('tip');
const colors = {role:'#ffffff',file:'#4e79a7',class:'#f28e2b',function:'#59a14f',state:'#e15759',config_arg:'#b07aa1',env:'#76b7b2',io:'#edc949',import:'#9c755f',external_call:'#777'};
for (const r of [...new Set(graph.nodes.map(n=>n.role).filter(Boolean))].sort()) { const o=document.createElement('option'); o.value=r; o.textContent=r; roleSel.appendChild(o); }
for (const k of [...new Set(graph.nodes.map(n=>n.kind))].sort()) { const o=document.createElement('option'); o.value=k; o.textContent=k; kindSel.appendChild(o); }
function edgeAllowed(e,m){ if(m==='agent_brief') return e.kind==='expected_loop'; if(m==='loop') return e.kind==='expected_loop'||e.kind==='role_contains'; if(m==='architecture') return e.kind==='role_contains'||e.kind==='expected_loop'; if(m==='calls') return ['calls','defines','owns'].includes(e.kind); if(m==='state') return ['reads_state','writes_state','defines','owns'].includes(e.kind); if(m==='io') return ['defines_arg','reads_env','reads_file','writes_file','defines'].includes(e.kind); if(m==='imports') return ['imports','defines'].includes(e.kind); return true; }
function filtered(){ let m=viewSel.value, edges=graph.edges.filter(e=>edgeAllowed(e,m)); let ids=new Set(); edges.forEach(e=>{ids.add(e.source);ids.add(e.target);}); let nodes=graph.nodes.filter(n=>ids.has(n.id));
 if(m==='agent_brief'){nodes=graph.nodes.filter(n=>n.kind==='role'); ids=new Set(nodes.map(n=>n.id)); edges=graph.edges.filter(e=>e.kind==='expected_loop'&&ids.has(e.source)&&ids.has(e.target));}
 if(m==='risk'){let riskIds=new Set((graph.analysis.risk_nodes||[]).slice(0,80).map(x=>x.id)); nodes=graph.nodes.filter(n=>riskIds.has(n.id)); ids=new Set(nodes.map(n=>n.id)); edges=graph.edges.filter(e=>ids.has(e.source)&&ids.has(e.target));}
 const rr=roleSel.value, kk=kindSel.value, qq=qInp.value.toLowerCase(); nodes=nodes.filter(n=>(!rr||n.role===rr)&&(!kk||n.kind===kk)&&(!qq||JSON.stringify(n).toLowerCase().includes(qq))); ids=new Set(nodes.map(n=>n.id)); edges=edges.filter(e=>ids.has(e.source)&&ids.has(e.target)); return {nodes,edges};}
function render(){svg.innerHTML=''; const W=svg.clientWidth||window.innerWidth,H=svg.clientHeight||(window.innerHeight-70); const {nodes,edges}=filtered(); stats.textContent=`${nodes.length} nodes / ${edges.length} edges`;
 const defs=document.createElementNS('http://www.w3.org/2000/svg','defs'); defs.innerHTML='<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#aaa"/></marker>'; svg.appendChild(defs);
 const pos={}, mode=viewSel.value;
 if(mode==='agent_brief'||mode==='loop'){ const roleNodes=nodes.filter(n=>n.kind==='role'),cx=W/2,cy=H/2,R=Math.max(100,Math.min(W,H)*0.32); roleNodes.forEach((n,i)=>{const a=-Math.PI/2+i*2*Math.PI/Math.max(1,roleNodes.length);pos[n.id]={x:cx+Math.cos(a)*R,y:cy+Math.sin(a)*R};}); nodes.filter(n=>n.kind!=='role').forEach((n,i)=>{const hub=nodes.find(x=>x.id===`role:${n.role}`);const hp=hub?pos[hub.id]:{x:cx,y:cy};const a=i*2.399;pos[n.id]={x:hp.x+Math.cos(a)*70,y:hp.y+Math.sin(a)*70};});}
 else { const by={}; nodes.forEach(n=>(by[n.kind]??=[]).push(n)); const kinds=Object.keys(by).sort(); kinds.forEach((k,ki)=>{const arr=by[k],x=(ki+1)*W/(kinds.length+1);arr.forEach((n,i)=>pos[n.id]={x:x,y:55+(i+1)*(H-110)/(arr.length+1)});});}
 nodes.forEach((n,i)=>{if(!pos[n.id])pos[n.id]={x:80+(i%10)*150,y:80+Math.floor(i/10)*50};});
 edges.forEach(e=>{const a=pos[e.source],b=pos[e.target]; if(!a||!b)return; const line=document.createElementNS('http://www.w3.org/2000/svg','line'); line.setAttribute('x1',a.x);line.setAttribute('y1',a.y);line.setAttribute('x2',b.x);line.setAttribute('y2',b.y);line.setAttribute('marker-end','url(#arrow)');line.setAttribute('stroke-width',e.kind==='expected_loop'?3:1); if(e.kind==='expected_loop')line.setAttribute('stroke','#20c997'); svg.appendChild(line);});
 nodes.forEach(n=>{const p=pos[n.id],g=document.createElementNS('http://www.w3.org/2000/svg','g'),c=document.createElementNS('http://www.w3.org/2000/svg','circle'),t=document.createElementNS('http://www.w3.org/2000/svg','text'); c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',n.kind==='role'?24:(n.kind==='file'?10:7));c.setAttribute('fill',colors[n.kind]||'#aaa');c.setAttribute('stroke',n.kind==='role'?'#20c997':'#222');c.setAttribute('stroke-width',n.kind==='role'?3:1); t.setAttribute('x',p.x+14);t.setAttribute('y',p.y+4);let label=n.label||n.id;if(label.length>44)label=label.slice(0,41)+'…';t.textContent=label;g.appendChild(c);g.appendChild(t);svg.appendChild(g);g.addEventListener('mousemove',ev=>{tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px';tip.innerHTML=`<b>${n.label}</b><br>kind=${n.kind}<br>role=${n.role||''}<br>file=${n.file||''}<br>line=${n.line||''}<br><small>${n.id}</small>`;});g.addEventListener('mouseleave',()=>tip.style.display='none');});
}
viewSel.onchange=render;roleSel.onchange=render;kindSel.onchange=render;qInp.oninput=render;document.getElementById('reset').onclick=()=>{viewSel.value='agent_brief';roleSel.value='';kindSel.value='';qInp.value='';render();};window.onresize=render;render();
</script></body></html>
"""

def write_dashboard(graph: dict[str, Any], path: Path) -> None:
    raw = json.dumps(graph, ensure_ascii=False).replace("</script>", "<\\/script>")
    path.write_text(HTML_TEMPLATE.replace("__GRAPH_JSON__", raw), encoding="utf-8")

def main() -> int:
    ap = argparse.ArgumentParser(description="Build adaptive code graph for agents.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()
    project_root = Path(args.root).resolve()
    cfg = load_config(project_root)
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = build_graph(project_root, cfg)
    (out_dir / "code_graph_full.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv_edges(graph, out_dir / "code_graph_edges.csv")
    write_agent_brief(graph, out_dir / "code_graph_agent_brief.md")
    write_hole_report(graph, out_dir / "code_graph_holes.md")
    write_dashboard(graph, out_dir / "code_graph_dashboard.html")
    print(f"[agent-codegraph] project={graph['project']}")
    print(f"[agent-codegraph] nodes={graph['summary']['nodes']} edges={graph['summary']['edges']} files={graph['summary']['files']} functions={graph['summary']['functions']}")
    for name, rep in graph["analysis"]["loop_reports"].items():
        print(f"[agent-codegraph] loop {name}: closed={rep['closed']} missing_roles={rep['missing_roles']} missing_edges={rep['missing_expected_edges']}")
    print(f"[agent-codegraph] wrote {out_dir / 'code_graph_dashboard.html'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
