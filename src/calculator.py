"""A restricted Python-expression interpreter, NOT an arbitrary Python sandbox.

No eval/exec, imports, attributes, subscripts, statements, files or network APIs.
Executed in a resource-limited child with an empty environment and private cwd.
"""
import ast
import json
import math
import operator
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Mod: operator.mod}
DEFAULT_LIMITS = {
    "timeout_seconds": 3, "cpu_seconds": 2, "memory_mb": 256, "open_files": 16,
    "max_expression_chars": 2000, "max_input_bytes": 100000, "max_ast_nodes": 200,
    "max_list_items": 1000, "max_abs_number": 1e100,
}
FUNCS = {"sum": sum, "mean": statistics.mean, "median": statistics.median,
         "min": min, "max": max, "len": len, "sqrt": math.sqrt, "abs": abs,
         "round": round}


def interpret(expression: str, data: dict, limits: dict | None = None) -> object:
    limits = DEFAULT_LIMITS | (limits or {})
    if len(expression) > limits["max_expression_chars"] or len(json.dumps(data).encode()) > limits["max_input_bytes"]:
        raise ValueError("Calculation input too large")
    for values in data.values():
        if not isinstance(values, list) or len(values) > limits["max_list_items"]:
            raise ValueError("Data list exceeds configured item limit")
        if any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > limits["max_abs_number"] for value in values):
            raise ValueError("Data number out of range")
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > limits["max_ast_nodes"]:
        raise ValueError("Expression too complex")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value) or abs(node.value) > limits["max_abs_number"]:
                raise ValueError("Number out of range")
            return node.value
        if isinstance(node, ast.List) and len(node.elts) <= limits["max_list_items"]:
            return [visit(v) for v in node.elts]
        if isinstance(node, ast.Name) and node.id in data:
            return data[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            a, b = visit(node.left), visit(node.right)
            if type(a) not in (int, float) or type(b) not in (int, float):
                raise ValueError("Binary operands must be numbers")
            return OPS[type(node.op)](a, b)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS and not node.keywords:
            return FUNCS[node.func.id](*(visit(a) for a in node.args))
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    result = visit(tree)
    numbers = result if isinstance(result, list) else [result]
    if any(not math.isfinite(value) or abs(value) > limits["max_abs_number"] for value in numbers):
        raise ValueError("Result number out of range")
    json.dumps(result, allow_nan=False)
    return result


def calculate(expression: str, data: dict, timeout: float | None = None, settings: dict | None = None) -> dict:
    limits = DEFAULT_LIMITS | (settings or {})
    payload = json.dumps({"expression": expression, "data": data})
    if len(payload.encode()) > limits["max_input_bytes"]:
        raise ValueError("Calculation input exceeds configured size limit")
    with tempfile.TemporaryDirectory(prefix="research-calc-") as cwd:
        completed = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--worker", json.dumps(limits)],
            input=payload, text=True,
            capture_output=True, timeout=limits["timeout_seconds"] if timeout is None else timeout, cwd=cwd, env={})
    if completed.returncode:
        raise ValueError("Calculation worker failed or exceeded resource limits")
    return json.loads(completed.stdout)


if __name__ == "__main__":
    import resource
    limits = DEFAULT_LIMITS | json.loads(sys.argv[2])
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"],) * 2)
    resource.setrlimit(resource.RLIMIT_AS, (limits["memory_mb"] * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (limits["open_files"],) * 2)
    try:
        args = json.loads(sys.stdin.read(limits["max_input_bytes"] + 1))
        print(json.dumps({"value": interpret(**args, limits=limits), "provenance": "calculated; verify input source citations"}))
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__ + ": " + str(exc)[:200]}))
