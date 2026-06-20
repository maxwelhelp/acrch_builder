from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

import torch
import yaml


TASK_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "tasks"
ALLOWED_PRIMITIVES = {
    "identity",
    "gated_keep",
    "diff",
    "contrast",
    "smooth",
    "low_rank",
    "channel",
    "ctx_matrix",
    "product",
    "gated_add",
    "merge",
    "split",
    "route",
    "edge_gate",
    "write_gate",
    "memory_read",
    "memory_write",
    "forget",
    "recall",
    "memory_gate",
    "output_write",
    "output_mix",
    "skip",
    "replace",
    "disable",
}


@dataclass(frozen=True)
class TaskConfig:
    name: str
    slots: int
    layers: int
    formula: str
    expected_actions: List[Dict[str, object]]
    pass_thresholds: Dict[str, Dict[str, float]]
    path: Path
    digest: str
    formula_tree: ast.Expression

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        values = {f"s{i}": x[:, i] for i in range(self.slots)}
        result = _eval_node(self.formula_tree.body, values)
        if not isinstance(result, torch.Tensor) or result.shape != (x.shape[0],):
            raise ValueError(
                f"formula must produce one scalar per sample; got {getattr(result, 'shape', None)}"
            )
        return result


def legacy_task_path(name: str) -> Path:
    path = TASK_CONFIG_DIR / f"{name}.yml"
    if not path.is_file():
        raise ValueError(f"unknown task: {name!r}; missing config {path}")
    return path


def load_task_config(path: str | Path) -> TaskConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"task config does not exist: {config_path}")
    raw_bytes = config_path.read_bytes()
    try:
        data = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("task config root must be a mapping")

    required = {"name", "slots", "layers", "formula", "expected_actions", "pass_thresholds"}
    unknown = set(data) - required
    missing = required - set(data)
    if missing or unknown:
        raise ValueError(f"task config schema mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")

    name = _nonempty_string(data["name"], "name")
    slots = _positive_int(data["slots"], "slots")
    layers = _positive_int(data["layers"], "layers")
    formula = _nonempty_string(data["formula"], "formula")
    tree = _validate_formula(formula, slots)
    actions = _validate_actions(data["expected_actions"], slots, layers)
    thresholds = _validate_thresholds(data["pass_thresholds"])
    digest = hashlib.sha256(raw_bytes).hexdigest()
    return TaskConfig(name, slots, layers, formula, actions, thresholds, config_path, digest, tree)


def evaluate_pass_thresholds(
    thresholds: Mapping[str, Mapping[str, float]], metrics: Mapping[str, object]
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for metric, limits in thresholds.items():
        value = metrics.get(metric)
        passed = isinstance(value, (int, float))
        if passed and "min" in limits:
            passed = float(value) >= limits["min"]
        if passed and "max" in limits:
            passed = float(value) <= limits["max"]
        results[metric] = {"value": value, **dict(limits), "passed": bool(passed)}
    return results


def _validate_formula(formula: str, slots: int) -> ast.Expression:
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid formula syntax: {exc.msg}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in {"mean", *(f"s{i}" for i in range(slots))}:
            raise ValueError(f"formula name is not allowed: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id != "mean" or len(node.args) != 1 or node.keywords:
                raise ValueError("only mean(expression) calls are allowed")
        elif not isinstance(
            node,
            (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant,
             ast.Add, ast.Sub, ast.Mult, ast.UAdd, ast.USub, ast.Call),
        ):
            raise ValueError(f"formula syntax is not allowed: {type(node).__name__}")
        if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or not isinstance(node.value, (int, float))):
            raise ValueError("formula constants must be numeric")
    return tree


def _eval_node(node: ast.AST, values: Mapping[str, torch.Tensor]):
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand, values)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _eval_node(node.left, values), _eval_node(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left * right
    if isinstance(node, ast.Call):
        return _eval_node(node.args[0], values).mean(dim=-1)
    raise AssertionError(f"validated AST contains unsupported node: {type(node).__name__}")


def _validate_actions(value: object, slots: int, layers: int) -> List[Dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError("expected_actions must be a non-empty list")
    result = []
    for index, action in enumerate(value):
        if not isinstance(action, Mapping) or set(action) != {"layer", "src", "tgt", "primitive"}:
            raise ValueError(f"expected_actions[{index}] must contain layer/src/tgt/primitive only")
        layer = _bounded_int(action["layer"], f"expected_actions[{index}].layer", layers)
        src = _bounded_int(action["src"], f"expected_actions[{index}].src", slots)
        tgt = _bounded_int(action["tgt"], f"expected_actions[{index}].tgt", slots)
        primitive = _nonempty_string(action["primitive"], f"expected_actions[{index}].primitive")
        if primitive not in ALLOWED_PRIMITIVES:
            raise ValueError(f"unsupported primitive in expected_actions[{index}]: {primitive}")
        result.append({"layer": layer, "src": src, "tgt": tgt, "primitive": primitive})
    return result


def _validate_thresholds(value: object) -> Dict[str, Dict[str, float]]:
    if not isinstance(value, Mapping):
        raise ValueError("pass_thresholds must be a mapping")
    result = {}
    for metric, limits in value.items():
        if not isinstance(metric, str) or not metric or not isinstance(limits, Mapping):
            raise ValueError("each pass threshold must map a metric name to min/max limits")
        if not limits or not set(limits) <= {"min", "max"}:
            raise ValueError(f"threshold {metric!r} must contain min and/or max only")
        parsed = {}
        for bound, number in limits.items():
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError(f"threshold {metric}.{bound} must be numeric")
            parsed[str(bound)] = float(number)
        result[metric] = parsed
    return result


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _bounded_int(value: object, field: str, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < upper:
        raise ValueError(f"{field} must be an integer in [0, {upper})")
    return value
