# Code graph pack v3

Offline code-logic graph builder. No D3, no CDN, no internet.

Install into repo root:

```bash
unzip -o ~/Загрузки/code_graph_pack_v3.zip -d .
chmod +x commands/build_code_logic_graph.sh
bash commands/build_code_logic_graph.sh
xdg-open reports/code_logic_graph.html
```

Outputs:
- reports/code_logic_graph.html
- reports/code_logic_graph.json
- reports/CODE_MAP.md
- reports/loop_closure_report.md

Views:
- architecture_loop
- architecture
- calls
- state
- io
- imports
- risk
