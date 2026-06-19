#!/usr/bin/env python3
from __future__ import annotations

import argparse, ast, html, json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
    "runs", "checkpoints", "weights", "data", "datasets", "wandb", "node_modules",
    "reports", "agent_reports",
}

ROLE_RULES = {
    "joint_controller": ["controller","choice","decision","actionmatrixlayer","action_matrix","mode_logits","choice_logits","gate","fanout","boundary"],
    "backbone": ["model","backbone","actionmatrixmodel","forward","state","slots","layer","activation"],
    "memory_lane": ["memory","mem","read","write","forget","recall","w_r","w_w","memory_read","memory_write"],
    "credit_collector": ["credit","ablation","delta_loss","real_gain","ema","collector","useful","suspicious"],
    "council": ["council","simulator","lowrank","sim","aux","predicted_gain","sim_quality"],
    "scanner": ["scanner","proposal","candidate","semantic","usage","topk","top_k","primitive search"],
    "primitive": ["primitive","primitives","primitive_matrix","primitiveinfo","descriptor","topology"],
    "executor": ["executor","execute","edge","write_gate","cell_mode","transform","skip","disable"],
    "report": ["report","metrics","trace","json","csv","html","writer"],
    "command": ["argparse","cli","main","run_","validate","command","bash"],
    "data": ["dataset","synthetic","sample","batch","loader","train","val"],
}
LOOP_ORDER = ["joint_controller","backbone","memory_lane","credit_collector","council"]
LOOP_EDGES = [
    ("joint_controller","backbone","coupled_decisions_per_step"),
    ("backbone","memory_lane","W_w_write"),
    ("memory_lane","joint_controller","W_r_read"),
    ("backbone","credit_collector","activations_delta_loss"),
    ("credit_collector","council","credit_signal_per_step"),
    ("council","joint_controller","calibrated_aux_loss_gradient"),
]

@dataclass
class FileInfo:
    path: str
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

def role_of(*parts: str) -> str:
    s = " ".join([p for p in parts if p]).lower()
    scores = Counter()
    for role, words in ROLE_RULES.items():
        for w in words:
            if w in s: scores[role] += 1
    return scores.most_common(1)[0][0] if scores else "unknown"

def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()

def should_skip(path: Path, root: Path, skip_dirs: set[str]) -> bool:
    try: rp = path.relative_to(root).as_posix()
    except ValueError: rp = path.as_posix()
    if set(Path(rp).parts) & skip_dirs: return True
    if rp == "tools/build_code_logic_graph.py": return True
    return False

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.Attribute):
        p = call_name(node.value)
        return f"{p}.{node.attr}" if p else node.attr
    if isinstance(node, ast.Call): return call_name(node.func)
    if isinstance(node, ast.Subscript): return call_name(node.value)
    return None

def lit(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

def self_attr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None

class FuncVisitor(ast.NodeVisitor):
    def __init__(self, info: FuncInfo): self.info = info
    def visit_Call(self, node: ast.Call) -> Any:
        name = call_name(node.func)
        if name: self.info.calls.append(name)
        if name and name.endswith("add_argument"):
            for a in node.args:
                s = lit(a)
                if s and s.startswith("--"): self.info.args.append(s)
        if name in {"os.getenv","os.environ.get","environ.get"} and node.args:
            s = lit(node.args[0])
            if s: self.info.env.append(s)
        if name == "open" or (name and name.endswith(".open")):
            mode = "r"
            if len(node.args) >= 2: mode = lit(node.args[1]) or mode
            for kw in node.keywords:
                if kw.arg == "mode": mode = lit(kw.value) or mode
            p = lit(node.args[0]) if node.args else "<dynamic_path>"
            (self.info.file_writes if any(x in mode for x in ["w","a","+"]) else self.info.file_reads).append(p or "<dynamic_path>")
        self.generic_visit(node)
    def visit_Attribute(self, node: ast.Attribute) -> Any:
        a = self_attr(node)
        if a: self.info.self_reads.append(a)
        self.generic_visit(node)
    def visit_Assign(self, node: ast.Assign) -> Any:
        for t in node.targets: self._target(t)
        self.visit(node.value)
    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        self._target(node.target)
        if node.value: self.visit(node.value)
    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        self._target(node.target); self.visit(node.value)
    def _target(self, node: ast.AST) -> None:
        a = self_attr(node)
        if a: self.info.self_writes.append(a)
        for c in ast.iter_child_nodes(node): self._target(c)

def parse_py(path: Path, root: Path) -> tuple[FileInfo, list[FuncInfo]]:
    text = read_text(path); rp = rel(path, root)
    fi = FileInfo(path=rp, lines=len(text.splitlines()))
    out: list[FuncInfo] = []
    try: tree = ast.parse(text, filename=rp)
    except SyntaxError:
        fi.role = "syntax_error"; return fi, out
    for n in ast.walk(tree):
        if isinstance(n, ast.Import): fi.imports.extend(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module: fi.imports.append(n.module)
        elif isinstance(n, ast.ClassDef): fi.classes.append(n.name)
    cls_stack: list[str] = []
    class V(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> Any:
            cls_stack.append(node.name); self.generic_visit(node); cls_stack.pop()
        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            cls = cls_stack[-1] if cls_stack else None
            label = f"{cls}.{node.name}" if cls else node.name
            info = FuncInfo(label=label, file=rp, line=node.lineno, class_name=cls)
            FuncVisitor(info).visit(node)
            for f in ["calls","self_reads","self_writes","args","env","file_reads","file_writes"]:
                setattr(info, f, sorted(set(getattr(info, f))))
            info.role = role_of(rp, label, " ".join(info.calls), " ".join(info.self_writes), " ".join(info.args))
            out.append(info); fi.functions.append(label)
        visit_AsyncFunctionDef = visit_FunctionDef
    V().visit(tree)
    fi.imports = sorted(set(fi.imports)); fi.classes = sorted(set(fi.classes)); fi.functions = sorted(set(fi.functions))
    fi.role = role_of(rp, " ".join(fi.classes), " ".join(fi.functions), " ".join(fi.imports))
    return fi, out

def project_files(root: Path, skip_dirs: set[str]) -> list[Path]:
    out=[]
    for p in root.rglob("*"):
        if p.is_file() and not should_skip(p, root, skip_dirs) and (p.suffix == ".py" or p.suffix in {".sh",".md"}):
            out.append(p)
    return sorted(out)

def build(root: Path, include_docs: bool, include_commands: bool, skip_dirs: set[str]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}; edges: list[dict[str, Any]] = []
    files: list[FileInfo] = []; funcs: list[FuncInfo] = []
    def node(i: str, label: str, kind: str, **kw: Any) -> None: nodes[i] = {"id":i,"label":label,"kind":kind,**kw}
    def edge(s: str, t: str, kind: str, **kw: Any) -> None:
        if s in nodes and t in nodes: edges.append({"source":s,"target":t,"kind":kind,**kw})
    for p in project_files(root, skip_dirs):
        rp=rel(p, root)
        if not include_docs and rp.startswith("docs/"): continue
        if not include_commands and rp.startswith("commands/"): continue
        if p.suffix == ".py": fi, fns = parse_py(p, root)
        else:
            text=read_text(p); fi, fns = FileInfo(path=rp, role=role_of(rp, text[:3000]), lines=len(text.splitlines())), []
        files.append(fi); funcs.extend(fns)
    by_simple: dict[str, list[str]] = defaultdict(list)
    for fn in funcs: by_simple[fn.label.split(".")[-1]].append(f"func:{fn.file}:{fn.label}")
    for fi in files:
        fid=f"file:{fi.path}"; node(fid, fi.path, "file", role=fi.role, lines=fi.lines)
        for imp in fi.imports:
            iid=f"import:{imp}"; node(iid, imp, "import", role="external"); edge(fid,iid,"imports")
        for c in fi.classes:
            cid=f"class:{fi.path}:{c}"; node(cid,c,"class",file=fi.path,role=role_of(fi.path,c)); edge(fid,cid,"defines")
    for fn in funcs:
        fid=f"func:{fn.file}:{fn.label}"; node(fid, fn.label, "function", file=fn.file, line=fn.line, role=fn.role)
        edge(f"file:{fn.file}", fid, "defines")
        if fn.class_name: edge(f"class:{fn.file}:{fn.class_name}", fid, "owns")
        for call in fn.calls:
            short=call.split(".")[-1]; targets=by_simple.get(short)
            if targets:
                for dst in targets[:3]:
                    if dst != fid: edge(fid,dst,"calls",call=call)
            elif call.startswith(("torch","F.","nn.","json","csv","Path","os.","subprocess")):
                eid=f"external:{call}"; node(eid,call,"external_call",role="external"); edge(fid,eid,"calls_external",call=call)
        for a in fn.self_reads:
            sid=f"state:{fn.file}:{fn.class_name or 'module'}:{a}"; node(sid,a,"state",file=fn.file,owner=fn.class_name,role=role_of(a)); edge(fid,sid,"reads_state")
        for a in fn.self_writes:
            sid=f"state:{fn.file}:{fn.class_name or 'module'}:{a}"; node(sid,a,"state",file=fn.file,owner=fn.class_name,role=role_of(a)); edge(fid,sid,"writes_state")
        for a in fn.args:
            aid=f"arg:{a}"; node(aid,a,"config_arg",role="command"); edge(fid,aid,"defines_arg")
        for evar in fn.env:
            eid=f"env:{evar}"; node(eid,evar,"env",role="command"); edge(fid,eid,"reads_env")
        for fr in fn.file_reads:
            rid=f"io_read:{fr}"; node(rid,fr,"io",role="report"); edge(fid,rid,"reads_file")
        for fw in fn.file_writes:
            wid=f"io_write:{fw}"; node(wid,fw,"io",role="report"); edge(fid,wid,"writes_file")
    role_members=defaultdict(list)
    for n in list(nodes.values()):
        r=n.get("role")
        if r and r not in {"unknown","external"}: role_members[r].append(n["id"])
    for r in sorted(role_members):
        rid=f"role:{r}"; node(rid,r,"role",role=r)
        for m in role_members[r]: edge(rid,m,"role_contains")
    missing_roles=[r for r in LOOP_ORDER if not role_members.get(r)]
    missing_edges=[]
    for a,b,label in LOOP_EDGES:
        if f"role:{a}" in nodes and f"role:{b}" in nodes:
            edges.append({"source":f"role:{a}","target":f"role:{b}","kind":"expected_loop","label":label})
        else: missing_edges.append(label)
    summary={"nodes":len(nodes),"edges":len(edges),"files":sum(1 for n in nodes.values() if n["kind"]=="file"),"functions":sum(1 for n in nodes.values() if n["kind"]=="function"),"roles":dict(Counter(n.get("role","unknown") for n in nodes.values())),"edge_kinds":dict(Counter(e["kind"] for e in edges)),"missing_roles":missing_roles,"missing_loop_edges":missing_edges}
    return {"schema":"code_logic_graph.v3.offline","summary":summary,"nodes":list(nodes.values()),"edges":edges}

def write_md(graph: dict[str, Any], path: Path) -> None:
    nodes=graph["nodes"]; edges=graph["edges"]; inc=Counter(e["target"] for e in edges); out=Counter(e["source"] for e in edges)
    risk=sorted(nodes, key=lambda n: inc[n["id"]]+out[n["id"]], reverse=True)[:40]
    s=graph["summary"]
    lines=["# CODE_MAP\n\n", f"- nodes: `{s['nodes']}`\n", f"- edges: `{s['edges']}`\n", f"- files: `{s['files']}`\n", f"- functions: `{s['functions']}`\n\n", "## Closed-loop check\n\n"]
    for r in LOOP_ORDER: lines.append(f"- `{r}`: `{sum(1 for n in nodes if n.get('role') == r)}` detected\n")
    lines += ["\n## Missing\n\n", f"- missing roles: `{s['missing_roles']}`\n", f"- missing loop edges: `{s['missing_loop_edges']}`\n\n", "## Views\n\n"]
    for v in ["architecture_loop","architecture","calls","state","io","imports","risk"]: lines.append(f"- `{v}`\n")
    lines.append("\n## Most connected edit-risk nodes\n\n")
    for n in risk:
        deg=inc[n["id"]]+out[n["id"]]
        lines.append(f"- `{n['label']}` kind=`{n['kind']}` role=`{n.get('role','')}` degree=`{deg}` file=`{n.get('file','')}`\n")
    path.write_text("".join(lines), encoding="utf-8")

def write_loop_report(graph: dict[str, Any], path: Path) -> None:
    s=graph["summary"]; lines=["# LOOP_CLOSURE_REPORT\n\nStatic check of the target closed learning contour.\n\n## Required blocks\n\n"]
    for r in LOOP_ORDER: lines.append(f"- `{r}`: `{sum(1 for n in graph['nodes'] if n.get('role') == r)}`\n")
    lines.append("\n## Missing concrete edges\n\n")
    if s["missing_loop_edges"]:
        for e in s["missing_loop_edges"]: lines.append(f"- `{e}`\n")
    else: lines.append("- none\n")
    lines.append("\nNote: this is static analysis. It proves the code has readable blocks/interfaces, not that training credit is mathematically correct.\n")
    path.write_text("".join(lines), encoding="utf-8")

HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Code Logic Graph v3 offline</title>
<style>
body{margin:0;background:#101014;color:#eee;font-family:system-ui,sans-serif}
#top{padding:10px 14px;border-bottom:1px solid #333;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
#g{width:100vw;height:calc(100vh - 64px);display:block}
select,input,button{background:#1e1e25;color:#eee;border:1px solid #444;border-radius:6px;padding:6px}
.tip{position:fixed;display:none;pointer-events:none;background:#222;border:1px solid #555;border-radius:8px;padding:8px;max-width:520px;font-size:12px;z-index:9}
line{stroke:#777;stroke-opacity:.35} text{fill:#eee;font-size:11px;pointer-events:none}.small{font-size:12px;color:#aaa}
</style></head><body>
<div id="top"><b>Code Logic Graph v3</b>
<label>view <select id="view"><option>architecture_loop</option><option>architecture</option><option>calls</option><option>state</option><option>io</option><option>imports</option><option>risk</option><option>all</option></select></label>
<label>role <select id="role"><option value="">all</option></select></label>
<label>kind <select id="kind"><option value="">all</option></select></label>
<input id="q" placeholder="search file/function/state"><button id="reset">reset</button><span id="stats" class="small"></span></div>
<svg id="g"></svg><div id="tip" class="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const graph = JSON.parse(document.getElementById('data').textContent);
const svg = document.getElementById('g'), view=document.getElementById('view'), role=document.getElementById('role'), kind=document.getElementById('kind'), q=document.getElementById('q'), stats=document.getElementById('stats'), tip=document.getElementById('tip');
const colors={role:'#ffffff',file:'#4e79a7',class:'#f28e2b',function:'#59a14f',state:'#e15759',config_arg:'#b07aa1',env:'#76b7b2',io:'#edc949',import:'#9c755f',external_call:'#777'};
for (const r of [...new Set(graph.nodes.map(n=>n.role).filter(Boolean))].sort()){const o=document.createElement('option');o.value=r;o.textContent=r;role.appendChild(o);}
for (const k of [...new Set(graph.nodes.map(n=>n.kind))].sort()){const o=document.createElement('option');o.value=k;o.textContent=k;kind.appendChild(o);}
function edgeAllowed(e,m){if(m==='architecture_loop')return e.kind==='expected_loop'||e.kind==='role_contains'; if(m==='architecture')return e.kind==='role_contains'||e.kind==='expected_loop'; if(m==='calls')return ['calls','defines','owns'].includes(e.kind); if(m==='state')return ['reads_state','writes_state','defines','owns'].includes(e.kind); if(m==='io')return ['defines_arg','reads_env','reads_file','writes_file','defines'].includes(e.kind); if(m==='imports')return ['imports','defines'].includes(e.kind); return true;}
function filtered(){let m=view.value, edges=graph.edges.filter(e=>edgeAllowed(e,m)); let ids=new Set(); edges.forEach(e=>{ids.add(e.source);ids.add(e.target);}); let nodes=graph.nodes.filter(n=>ids.has(n.id));
 if(m==='architecture_loop')nodes=nodes.filter(n=>n.kind==='role'||['joint_controller','backbone','memory_lane','credit_collector','council'].includes(n.role));
 if(m==='risk'){const deg={}; graph.edges.forEach(e=>{deg[e.source]=(deg[e.source]||0)+1;deg[e.target]=(deg[e.target]||0)+1;}); nodes=[...graph.nodes].sort((a,b)=>(deg[b.id]||0)-(deg[a.id]||0)).slice(0,80); ids=new Set(nodes.map(n=>n.id)); edges=graph.edges.filter(e=>ids.has(e.source)&&ids.has(e.target));}
 const rr=role.value,kk=kind.value,qq=q.value.toLowerCase(); nodes=nodes.filter(n=>(!rr||n.role===rr)&&(!kk||n.kind===kk)&&(!qq||JSON.stringify(n).toLowerCase().includes(qq))); ids=new Set(nodes.map(n=>n.id)); edges=edges.filter(e=>ids.has(e.source)&&ids.has(e.target)); return {nodes,edges};}
function render(){svg.innerHTML=''; const W=svg.clientWidth||window.innerWidth,H=svg.clientHeight||(window.innerHeight-64); const {nodes,edges}=filtered(); stats.textContent=`${nodes.length} nodes / ${edges.length} edges`;
 const defs=document.createElementNS('http://www.w3.org/2000/svg','defs'); defs.innerHTML='<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#999"/></marker>'; svg.appendChild(defs);
 const pos={}, m=view.value;
 if(m==='architecture_loop'){const loop=['joint_controller','backbone','credit_collector','council','memory_lane'],cx=W/2,cy=H/2,R=Math.min(W,H)*0.31; for(let i=0;i<loop.length;i++){const a=-Math.PI/2+i*2*Math.PI/loop.length; for(const n of nodes.filter(n=>n.id===`role:${loop[i]}`||n.role===loop[i])){const jitter=n.kind==='role'?0:45; pos[n.id]={x:cx+Math.cos(a)*(R+jitter),y:cy+Math.sin(a)*(R+jitter)};}}}
 else{const by={};nodes.forEach(n=>(by[n.kind]??=[]).push(n)); const ks=Object.keys(by).sort(); ks.forEach((k,ki)=>{const arr=by[k],x=(ki+1)*W/(ks.length+1); arr.forEach((n,i)=>pos[n.id]={x:x,y:60+(i+1)*(H-120)/(arr.length+1)});});}
 nodes.forEach((n,i)=>{if(!pos[n.id])pos[n.id]={x:70+(i%12)*125,y:80+Math.floor(i/12)*42};});
 edges.forEach(e=>{const a=pos[e.source],b=pos[e.target]; if(!a||!b)return; const line=document.createElementNS('http://www.w3.org/2000/svg','line'); line.setAttribute('x1',a.x);line.setAttribute('y1',a.y);line.setAttribute('x2',b.x);line.setAttribute('y2',b.y);line.setAttribute('stroke-width',e.kind==='expected_loop'?3:1);line.setAttribute('marker-end','url(#arrow)'); if(e.kind==='expected_loop')line.setAttribute('stroke','#20c997'); svg.appendChild(line);});
 nodes.forEach(n=>{const p=pos[n.id],g=document.createElementNS('http://www.w3.org/2000/svg','g'),c=document.createElementNS('http://www.w3.org/2000/svg','circle'),t=document.createElementNS('http://www.w3.org/2000/svg','text'); c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',n.kind==='role'?20:(n.kind==='file'?10:7));c.setAttribute('fill',colors[n.kind]||'#aaa');c.setAttribute('stroke',n.kind==='role'?'#20c997':'#222');c.setAttribute('stroke-width',n.kind==='role'?3:1); t.setAttribute('x',p.x+12);t.setAttribute('y',p.y+4);let label=n.label||n.id;if(label.length>42)label=label.slice(0,39)+'…';t.textContent=label;g.appendChild(c);g.appendChild(t);svg.appendChild(g);g.addEventListener('mousemove',ev=>{tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px';tip.innerHTML=`<b>${n.label}</b><br>kind=${n.kind}<br>role=${n.role||''}<br>file=${n.file||''}<br>line=${n.line||''}<br><small>${n.id}</small>`;});g.addEventListener('mouseleave',()=>tip.style.display='none');});
}
view.onchange=render;role.onchange=render;kind.onchange=render;q.oninput=render;document.getElementById('reset').onclick=()=>{view.value='architecture_loop';role.value='';kind.value='';q.value='';render();};window.onresize=render;render();
</script></body></html>"""

def write_html(graph: dict[str, Any], path: Path) -> None:
    path.write_text(HTML_TEMPLATE.replace("__DATA__", html.escape(json.dumps(graph, ensure_ascii=False))), encoding="utf-8")

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out-json", default="reports/code_logic_graph.json")
    ap.add_argument("--out-html", default="reports/code_logic_graph.html")
    ap.add_argument("--out-md", default="reports/CODE_MAP.md")
    ap.add_argument("--out-loop", default="reports/loop_closure_report.md")
    ap.add_argument("--include-docs", action="store_true")
    ap.add_argument("--include-commands", action="store_true")
    ap.add_argument("--include-reports", action="store_true")
    args=ap.parse_args()
    root=Path(args.root).resolve()
    skip_dirs=set(DEFAULT_SKIP_DIRS)
    if args.include_reports:
        skip_dirs.discard("reports"); skip_dirs.discard("agent_reports")
    graph=build(root, args.include_docs, args.include_commands, skip_dirs)
    for p in [args.out_json,args.out_html,args.out_md,args.out_loop]: (root/p).parent.mkdir(parents=True, exist_ok=True)
    (root/args.out_json).write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    write_html(graph, root/args.out_html); write_md(graph, root/args.out_md); write_loop_report(graph, root/args.out_loop)
    print(f"[code-graph-v3] nodes={graph['summary']['nodes']} edges={graph['summary']['edges']} files={graph['summary']['files']} functions={graph['summary']['functions']}")
    print(f"[code-graph-v3] missing_roles={graph['summary']['missing_roles']}")
    print(f"[code-graph-v3] missing_loop_edges={graph['summary']['missing_loop_edges']}")
    print(f"[code-graph-v3] wrote {args.out_html}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
