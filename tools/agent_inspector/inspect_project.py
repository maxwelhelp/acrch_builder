#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def rel(p: Path, root: Path) -> str:
    return p.resolve().relative_to(root.resolve()).as_posix()

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def role_of(cfg: dict[str, Any], *parts: str) -> str:
    s = " ".join(x for x in parts if x).lower()
    scores = Counter()
    for role, kws in cfg.get("roles", {}).items():
        for kw in kws:
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
    return None

def self_attr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None

class FV(ast.NodeVisitor):
    def __init__(self):
        self.calls: list[str] = []
        self.self_reads: list[str] = []
        self.self_writes: list[str] = []
        self.args: list[str] = []

    def visit_Call(self, node: ast.Call):
        name = call_name(node.func)
        if name:
            self.calls.append(name)
        if name and name.endswith("add_argument"):
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--"):
                    self.args.append(a.value)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        a = self_attr(node)
        if a:
            self.self_reads.append(a)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for t in node.targets:
            self._target(t)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self._target(node.target)
        if node.value:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign):
        self._target(node.target)
        self.visit(node.value)

    def _target(self, node: ast.AST):
        a = self_attr(node)
        if a:
            self.self_writes.append(a)
        for c in ast.iter_child_nodes(node):
            self._target(c)

def build_static(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    skip_dirs = set(cfg.get("static", {}).get("skip_dirs", []))
    exts = set(cfg.get("static", {}).get("code_extensions", [".py", ".sh", ".md", ".json"]))
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_node(i: str, label: str, kind: str, **kw):
        nodes[i] = {"id": i, "label": label, "kind": kind, **kw}

    def add_edge(s: str, t: str, kind: str, **kw):
        if s in nodes and t in nodes:
            edges.append({"source": s, "target": t, "kind": kind, **kw})

    py_funcs: dict[str, list[str]] = defaultdict(list)
    pending_calls: list[tuple[str, str]] = []

    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix not in exts:
            continue
        rp = rel(p, root)
        if set(Path(rp).parts) & skip_dirs:
            continue

        text = read_text(p)
        fid = f"file:{rp}"
        add_node(fid, rp, "file", role=role_of(cfg, rp, text[:2000]), lines=len(text.splitlines()))

        if p.suffix != ".py":
            continue

        try:
            tree = ast.parse(text, filename=rp)
        except SyntaxError:
            nodes[fid]["role"] = "syntax_error"
            continue

        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    iid = f"import:{a.name}"
                    add_node(iid, a.name, "import", role="external")
                    add_edge(fid, iid, "imports")
            elif isinstance(n, ast.ImportFrom) and n.module:
                iid = f"import:{n.module}"
                add_node(iid, n.module, "import", role="external")
                add_edge(fid, iid, "imports")
            elif isinstance(n, ast.ClassDef):
                cid = f"class:{rp}:{n.name}"
                add_node(cid, n.name, "class", role=role_of(cfg, rp, n.name), file=rp, line=n.lineno)
                add_edge(fid, cid, "defines")

        class_stack: list[str] = []
        class V(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef):
                class_stack.append(node.name)
                self.generic_visit(node)
                class_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef):
                cls = class_stack[-1] if class_stack else None
                label = f"{cls}.{node.name}" if cls else node.name
                vid = f"func:{rp}:{label}"
                fv = FV()
                fv.visit(node)
                role = role_of(cfg, rp, label, " ".join(fv.calls), " ".join(fv.self_writes), " ".join(fv.args))
                add_node(vid, label, "function", role=role, file=rp, line=node.lineno)
                add_edge(fid, vid, "defines")
                if cls:
                    add_edge(f"class:{rp}:{cls}", vid, "owns")
                py_funcs[node.name].append(vid)
                for c in set(fv.calls):
                    pending_calls.append((vid, c))
                for a in set(fv.self_reads):
                    sid = f"state:{rp}:{cls or 'module'}:{a}"
                    add_node(sid, a, "state", role=role_of(cfg, a), file=rp)
                    add_edge(vid, sid, "reads_state")
                for a in set(fv.self_writes):
                    sid = f"state:{rp}:{cls or 'module'}:{a}"
                    add_node(sid, a, "state", role=role_of(cfg, a), file=rp)
                    add_edge(vid, sid, "writes_state")
                for a in set(fv.args):
                    aid = f"arg:{a}"
                    add_node(aid, a, "config_arg", role="entrypoint")
                    add_edge(vid, aid, "defines_arg")

            visit_AsyncFunctionDef = visit_FunctionDef

        V().visit(tree)

    for src, call in pending_calls:
        short = call.split(".")[-1]
        for dst in py_funcs.get(short, [])[:4]:
            if src != dst:
                add_edge(src, dst, "calls", call=call)

    role_members: dict[str, list[str]] = defaultdict(list)
    for n in list(nodes.values()):
        r = n.get("role")
        if r and r not in {"unknown", "external", "syntax_error"}:
            role_members[r].append(n["id"])

    for role, members in role_members.items():
        rid = f"role:{role}"
        add_node(rid, role, "role", role=role)
        for m in members:
            add_edge(rid, m, "role_contains")

    for loop_name, roles in cfg.get("loops", {}).items():
        for a, b in zip(roles, roles[1:]):
            if f"role:{a}" in nodes and f"role:{b}" in nodes:
                edges.append({"source": f"role:{a}", "target": f"role:{b}", "kind": "expected_loop", "loop": loop_name, "label": f"{a}->{b}"})

    graph = {"schema": "static_code_graph.v1", "project": cfg.get("project_name", root.name), "nodes": list(nodes.values()), "edges": edges}
    graph["summary"] = {
        "nodes": len(graph["nodes"]),
        "edges": len(edges),
        "roles": dict(Counter(n.get("role", "unknown") for n in graph["nodes"])),
        "kinds": dict(Counter(n.get("kind") for n in graph["nodes"])),
        "edge_kinds": dict(Counter(e.get("kind") for e in edges)),
    }
    graph["analysis"] = analyze_static(graph, cfg)
    return graph

def analyze_static(graph: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    inc = Counter(e["target"] for e in edges)
    out = Counter(e["source"] for e in edges)
    loops = {}
    for name, roles in cfg.get("loops", {}).items():
        missing_roles = [r for r in roles if f"role:{r}" not in nodes]
        missing_edges = []
        for a, b in zip(roles, roles[1:]):
            if not any(e["kind"] == "expected_loop" and e["source"] == f"role:{a}" and e["target"] == f"role:{b}" for e in edges):
                missing_edges.append(f"{a}->{b}")
        loops[name] = {"roles": roles, "closed_static": not missing_roles and not missing_edges, "missing_roles": missing_roles, "missing_edges": missing_edges}
    risk = sorted(nodes.values(), key=lambda n: inc[n["id"]] + out[n["id"]], reverse=True)[:40]
    return {
        "loops": loops,
        "risk_nodes": [{"id": n["id"], "label": n["label"], "kind": n["kind"], "role": n.get("role"), "degree": inc[n["id"]] + out[n["id"]], "file": n.get("file", ""), "line": n.get("line", "")} for n in risk]
    }

def grad_graph(cfg: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    grad_norms = probe.get("grad_norms", {}) or {}
    critical = cfg.get("runtime_contract", {}).get("critical_grad_groups", [])
    optional = cfg.get("runtime_contract", {}).get("optional_grad_groups", [])
    min_g = float(cfg.get("runtime_contract", {}).get("min_grad_norm", 1e-12))
    nodes = [{"id": "loss", "label": "loss", "kind": "loss", "status": "source"}]
    edges = []
    bad = []
    for group in critical + optional:
        val = grad_norms.get(group)
        if val is None:
            status = "not_measured"
        elif float(val) > min_g:
            status = "grad_ok"
        else:
            status = "grad_zero"
        if group in critical and status != "grad_ok":
            bad.append(group)
        nodes.append({"id": group, "label": group, "kind": "grad_group", "grad_norm": val, "critical": group in critical, "status": status})
        edges.append({"source": "loss", "target": group, "kind": "gradient_path", "status": status})
    return {
        "schema": "runtime_gradient_graph.v1",
        "nodes": nodes,
        "edges": edges,
        "summary": {"gradient_closed": len(bad) == 0 and bool(critical), "bad_critical_groups": bad}
    }

def credit_graph(cfg: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    cp = probe.get("credit_path", {}) or {}
    order = cfg.get("runtime_contract", {}).get("credit_path_order", [])
    nodes = []
    bad = []
    for key in order:
        value = cp.get(key)
        if isinstance(value, bool):
            ok = value
        elif isinstance(value, (int, float)):
            if key == "choice_mass":
                ok = value > float(cfg["runtime_contract"].get("min_choice_mass", 1e-3))
            elif key == "edge_active":
                ok = value > float(cfg["runtime_contract"].get("min_edge_active", 1e-3))
            elif key == "recovery":
                ok = value > float(cfg["runtime_contract"].get("min_recovery", 1e-3))
            else:
                ok = value > 0
        else:
            ok = False
        status = "ok" if ok else "bad"
        if not ok:
            bad.append(key)
        nodes.append({"id": key, "label": key, "kind": "credit_stage", "value": value, "status": status})
    edges = []
    broken = []
    for a, b in zip(order, order[1:]):
        sa = next((n["status"] for n in nodes if n["id"] == a), "bad")
        sb = next((n["status"] for n in nodes if n["id"] == b), "bad")
        status = "ok" if sa == "ok" and sb == "ok" else "broken"
        if status != "ok":
            broken.append(f"{a}->{b}")
        edges.append({"source": a, "target": b, "kind": "credit_flow", "status": status})
    return {
        "schema": "runtime_credit_graph.v1",
        "nodes": nodes,
        "edges": edges,
        "summary": {"credit_closed": len(bad) == 0 and bool(order), "bad_stages": bad, "broken_edges": broken}
    }

def write_md(static: dict[str, Any], probe: dict[str, Any], gg: dict[str, Any], cg: dict[str, Any], out: Path):
    lines = ["# AGENT_CONTEXT\n\n"]
    lines.append("## Runtime truth\n\n")
    lines.append(f"- gradient_closed: `{gg['summary']['gradient_closed']}` bad=`{gg['summary']['bad_critical_groups']}`\n")
    lines.append(f"- credit_closed: `{cg['summary']['credit_closed']}` bad=`{cg['summary']['bad_stages']}` broken=`{cg['summary']['broken_edges']}`\n\n")
    health = probe.get("health", {}) or {}
    lines.append(f"- recovery_loss_connected: `{health.get('recovery_loss_connected')}`\n")
    lines.append(f"- detached_enabled_losses: `{health.get('detached_enabled_losses', [])}`\n\n")
    lines.append("## Static loops\n\n")
    for name, rep in static["analysis"]["loops"].items():
        lines.append(f"- `{name}` static_closed=`{rep['closed_static']}` missing_roles=`{rep['missing_roles']}` missing_edges=`{rep['missing_edges']}`\n")
    lines.append("\n## Risk nodes\n\n")
    for n in static["analysis"]["risk_nodes"][:25]:
        lines.append(f"- `{n['label']}` role=`{n['role']}` degree=`{n['degree']}` file=`{n['file']}` line=`{n['line']}`\n")
    lines.append("\n## Probe metrics\n\n")
    for k, v in (probe.get("metrics", {}) or {}).items():
        lines.append(f"- `{k}`: `{v}`\n")
    lines.append("\n## Loss connectivity\n\n")
    for name, item in (probe.get("loss_connectivity", {}) or {}).items():
        lines.append(
            f"- `{name}`: requires_grad=`{item.get('requires_grad')}` "
            f"applicable=`{item.get('applicable')}` "
            f"connected=`{item.get('connected_to_intended_target')}` "
            f"target_grad_norms=`{item.get('target_grad_norms', {})}`\n"
        )
    out.write_text("".join(lines), encoding="utf-8")

def write_dashboard(static: dict[str, Any], out: Path):
    raw = json.dumps(static, ensure_ascii=False).replace("</script>", "<\\/script>")
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Inspector</title>
<style>body{{margin:0;background:#111;color:#eee;font-family:system-ui}}#top{{padding:10px;border-bottom:1px solid #333;display:flex;gap:10px}}svg{{width:100vw;height:calc(100vh - 55px)}}text{{fill:#eee;font-size:11px}}line{{stroke:#888;stroke-opacity:.4}}</style></head><body>
<div id='top'><b>Static graph</b><select id='view'><option>loops</option><option>roles</option><option>risk</option><option>full</option></select><span id='stats'></span></div><svg id='svg'></svg>
<script id='data' type='application/json'>{raw}</script><script>
const g=JSON.parse(document.getElementById('data').textContent),svg=document.getElementById('svg'),view=document.getElementById('view'),stats=document.getElementById('stats');
function allowed(e,m){{if(m==='loops')return e.kind==='expected_loop';if(m==='roles')return e.kind==='expected_loop'||e.kind==='role_contains';if(m==='risk')return true;return true}}
function render(){{svg.innerHTML='';let m=view.value,edges=g.edges.filter(e=>allowed(e,m)),ids=new Set();edges.forEach(e=>{{ids.add(e.source);ids.add(e.target)}});let nodes=g.nodes.filter(n=>ids.has(n.id));if(m==='risk'){{ids=new Set(g.analysis.risk_nodes.slice(0,60).map(x=>x.id));nodes=g.nodes.filter(n=>ids.has(n.id));edges=g.edges.filter(e=>ids.has(e.source)&&ids.has(e.target))}}stats.textContent=`${{nodes.length}} nodes / ${{edges.length}} edges`;let W=svg.clientWidth||innerWidth,H=svg.clientHeight||innerHeight-55,pos={{}};let by={{}};nodes.forEach(n=>(by[n.kind]??=[]).push(n));Object.keys(by).sort().forEach((k,ki,arr)=>{{let x=(ki+1)*W/(arr.length+1);by[k].forEach((n,i)=>pos[n.id]={{x:x,y:50+(i+1)*(H-100)/(by[k].length+1)}})}});edges.forEach(e=>{{let a=pos[e.source],b=pos[e.target];if(!a||!b)return;let l=document.createElementNS('http://www.w3.org/2000/svg','line');l.setAttribute('x1',a.x);l.setAttribute('y1',a.y);l.setAttribute('x2',b.x);l.setAttribute('y2',b.y);if(e.kind==='expected_loop'){{l.setAttribute('stroke','#20c997');l.setAttribute('stroke-width','3')}}svg.appendChild(l)}});nodes.forEach(n=>{{let p=pos[n.id],c=document.createElementNS('http://www.w3.org/2000/svg','circle'),t=document.createElementNS('http://www.w3.org/2000/svg','text');c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',n.kind==='role'?18:7);c.setAttribute('fill',n.kind==='role'?'#fff':n.kind==='function'?'#59a14f':n.kind==='file'?'#4e79a7':'#999');svg.appendChild(c);t.setAttribute('x',p.x+10);t.setAttribute('y',p.y+4);t.textContent=(n.label||n.id).slice(0,45);svg.appendChild(t)}})}}
view.onchange=render;render();
</script></body></html>"""
    out.write_text(html, encoding="utf-8")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    cfg = load_json(root / ".inspector.json", {})
    out_dir = root / cfg.get("paths", {}).get("out_dir", "reports/agent_inspector")
    out_dir.mkdir(parents=True, exist_ok=True)

    static = build_static(root, cfg)
    probe = load_json(root / cfg.get("paths", {}).get("runtime_probe", "reports/agent_inspector/runtime_probe.json"), {"metrics": {}, "grad_norms": {}, "credit_path": {}, "health": {}})
    gg = grad_graph(cfg, probe)
    cg = credit_graph(cfg, probe)

    (out_dir / "static_graph.json").write_text(json.dumps(static, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "runtime_gradient_graph.json").write_text(json.dumps(gg, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "runtime_credit_graph.json").write_text(json.dumps(cg, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "runtime_probe_loaded.json").write_text(json.dumps(probe, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(static, probe, gg, cg, out_dir / "AGENT_CONTEXT.md")
    write_dashboard(static, out_dir / "static_dashboard.html")

    with (out_dir / "static_edges.csv").open("w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["source", "target", "kind", "label", "loop", "call"])
        w.writeheader()
        for e in static["edges"]:
            w.writerow({k: e.get(k, "") for k in ["source", "target", "kind", "label", "loop", "call"]})

    summary = [
        "# Inspector Summary\n\n",
        f"- gradient_closed: `{gg['summary']['gradient_closed']}`\n",
        f"- credit_closed: `{cg['summary']['credit_closed']}`\n",
        f"- bad_grad_groups: `{gg['summary']['bad_critical_groups']}`\n",
        f"- bad_credit_stages: `{cg['summary']['bad_stages']}`\n",
        f"- recovery_loss_connected: `{(probe.get('health', {}) or {}).get('recovery_loss_connected')}`\n",
        f"- detached_enabled_losses: `{(probe.get('health', {}) or {}).get('detached_enabled_losses', [])}`\n",
    ]
    (out_dir / "SUMMARY.md").write_text("".join(summary), encoding="utf-8")

    print(f"[inspect] wrote {out_dir}")
    print(f"[inspect] gradient_closed={gg['summary']['gradient_closed']} bad={gg['summary']['bad_critical_groups']}")
    print(f"[inspect] credit_closed={cg['summary']['credit_closed']} bad={cg['summary']['bad_stages']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
