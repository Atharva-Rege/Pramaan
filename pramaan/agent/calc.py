from __future__ import annotations

import ast
import operator
from typing import Any

from pramaan.core.errors import ToolError

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_LEN = 300
MAX_NODES = 80


def _eval(node: ast.AST, env: dict[str, float], count: list[int]) -> float:
    count[0] += 1
    if count[0] > MAX_NODES:
        raise ToolError("expression too complex")
    if isinstance(node, ast.Expression):
        return _eval(node.body, env, count)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("only numeric constants are allowed")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ToolError(f"unknown variable {node.id}")
        return float(env[node.id])
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _eval(node.left, env, count), _eval(node.right, env, count)
        if isinstance(node.op, ast.Div) and b == 0:
            raise ToolError("division by zero")
        return float(_BIN[type(node.op)](a, b))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return float(_UN[type(node.op)](_eval(node.operand, env, count)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        if node.func.id == "abs" and len(node.args) == 1:
            return abs(_eval(node.args[0], env, count))
        if node.func.id == "round" and 1 <= len(node.args) <= 2:
            x = _eval(node.args[0], env, count)
            nd = int(_eval(node.args[1], env, count)) if len(node.args) == 2 else 0
            return float(round(x, nd))
    raise ToolError(f"unsupported syntax: {type(node).__name__}")


def calculate(expression: str, variables: dict[str, Any] | None = None) -> float:
    if len(expression) > MAX_LEN:
        raise ToolError("expression too long")
    env: dict[str, float] = {}
    for k, v in (variables or {}).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ToolError(f"variable {k} is not numeric")
        env[str(k)] = float(v)
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"syntax error: {exc.msg}") from exc
    return _eval(tree, env, [0])
