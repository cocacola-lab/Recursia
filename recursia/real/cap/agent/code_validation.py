# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast, builtins
_IMPORTS = {"numpy", "math", "scipy.spatial.transform"}

def validate_code(code, *, allow_globals=False):
    if not code.strip() or len(code) > 50000:
        raise ValueError("empty or excessively long generated policy")
    tree = ast.parse(code)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    forbidden = {"open", "exec", "eval", "compile", "globals", "locals",
                 "setattr", "delattr", "vars", "input", "breakpoint", "__import__"}
    if allow_globals:
        forbidden.remove("globals")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            if getattr(node, "level", 0) or any(m not in _IMPORTS for m in modules):
                raise ValueError("policy imports are limited to numpy, math, scipy.spatial.transform")
        if isinstance(node, ast.Name) and (node.id in forbidden or node.id.startswith("__")):
            raise ValueError(f"unavailable policy name {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            type_name = (node.attr == "__name__" and isinstance(node.value, ast.Call)
                         and isinstance(node.value.func, ast.Name) and node.value.func.id == "type"
                         and len(node.value.args) == 1 and not node.value.keywords)
            class_name = (node.attr == "__name__" and isinstance(node.value, ast.Attribute)
                          and node.value.attr == "__class__")
            class_for_name = (node.attr == "__class__" and isinstance(parents.get(node), ast.Attribute)
                              and parents[node].attr == "__name__")
            if not (type_name or class_name or class_for_name):
                raise ValueError(f"private attribute .{node.attr} at line {node.lineno} is unavailable; "
                                 "only type(error).__name__ / error.__class__.__name__ logging is allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {
            "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "fromfile", "tofile", "memmap"}:
            raise ValueError("policy file I/O is unavailable")
    return tree

def _policy_import(name, globals=None, locals=None, fromlist=(), level=0):
    # ndarray.min/max and some other NumPy C methods dynamically import their
    # own implementation through the caller's __import__ hook.
    if level or not (name in _IMPORTS or name.startswith(("numpy.", "scipy."))):
        raise ImportError(f"policy import not available: {name}")
    return builtins.__import__(name, globals, locals, fromlist, level)
