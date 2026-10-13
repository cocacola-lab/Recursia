"""Restrict robot access to the six primitives, including learned programs."""
import ast
import builtins
import math
from types import MappingProxyType

import numpy as np
from scipy.spatial.transform import Rotation

from cap.agent.code_validation import _policy_import, validate_code
from .prompts import TOOLS

BUILTINS = ("abs", "all", "any", "bool", "dict", "enumerate", "Exception", "float", "int", "isinstance",
            "len", "list", "map", "max", "min", "range", "reversed", "round", "RuntimeError", "set",
            "sorted", "str", "sum", "tuple", "ValueError", "zip", "KeyError", "TypeError",
            "IndexError", "ImportError", "AssertionError", "StopIteration", "type", "getattr",
            "globals", "NameError", "UnboundLocalError")

def _diagnostic_literals(tree):
    """Exempt direct diagnostic text, without exempting expressions inside it."""
    literals = set()
    diagnostic_names = set()

    def text_parts(value):
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            literals.add(value)
        elif isinstance(value, ast.Name):
            diagnostic_names.add(value)
        elif isinstance(value, ast.JoinedStr):
            # Interpolated expressions can still contain perception calls or
            # task-specific branches; only the surrounding text is diagnostic.
            for part in value.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    literals.add(part)

    messages = {"reason", "message", "note", "diagnosis", "detail", "description"}
    exceptions = {"Exception", "RuntimeError", "ValueError", "KeyError", "TypeError",
                  "IndexError", "ImportError", "AssertionError", "StopIteration", "NameError", "UnboundLocalError"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    text_parts(value)  # Python docstring, never an executed action.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            for value in node.args:
                text_parts(value)
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            call = node.exc
            if isinstance(call.func, ast.Name) and call.func.id in exceptions:
                for value in call.args:
                    text_parts(value)
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in messages:
                    text_parts(value)

    # A helper such as _unverified(reason) often forwards a message into a
    # diagnostic field. Infer its diagnostic parameters from their uses,
    # rather than trusting the function or parameter name.
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    while True:
        before = (len(literals), len(diagnostic_names))
        for helper in helpers:
            parameters = [*helper.args.posonlyargs, *helper.args.args, *helper.args.kwonlyargs]
            diagnostic_parameters = set()
            for parameter in parameters:
                uses = [node for node in ast.walk(helper) if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load) and node.id == parameter.arg]
                if uses and all(node in diagnostic_names for node in uses):
                    diagnostic_parameters.add(parameter.arg)
            positional = [*helper.args.posonlyargs, *helper.args.args]
            for call in ast.walk(tree):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == helper.name):
                    continue
                for parameter, value in zip(positional, call.args):
                    if parameter.arg in diagnostic_parameters:
                        text_parts(value)
                for keyword in call.keywords:
                    if keyword.arg in diagnostic_parameters:
                        text_parts(keyword.value)
        if before == (len(literals), len(diagnostic_names)):
            break
    return literals

def validate_policy(code, *, stack=False, forbidden_literals=(), known_names=()):
    tree = validate_code(code, allow_globals=True)
    allowed = set(known_names) | set(TOOLS) | set(BUILTINS) | {"np", "Rotation", "math", "TASK", "print", "numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            allowed.add(node.id)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in TOOLS:
                raise ValueError("policy cannot redefine a primitive tool")
            if isinstance(node, ast.AsyncFunctionDef):
                raise ValueError("async policy functions are unavailable")
            allowed.add(node.name)
            allowed.update(a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs)
            if node.args.vararg:
                allowed.add(node.args.vararg.arg)
            if node.args.kwarg:
                allowed.add(node.args.kwarg.arg)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in TOOLS:
            raise ValueError("policy cannot replace a primitive tool")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            allowed.update(a.asname or a.name.split(".")[0] for a in node.names)
        if isinstance(node, ast.ExceptHandler) and node.name:
            allowed.add(node.name)
    def check_names(node, lambda_names=frozenset()):
        if isinstance(node, ast.Lambda):
            # Defaults use the enclosing scope; parameters are local to the
            # body, including nested lambdas that close over them.
            for value in (*node.args.defaults, *node.args.kw_defaults):
                if value is not None:
                    check_names(value, lambda_names)
            parameters = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
            parameters.update(arg.arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None)
            check_names(node.body, lambda_names | parameters)
            return
        # A variable can be created by a previously defined helper, or probed
        # with try/except NameError. Let Python resolve data names at execution
        # time; an unknown direct function/tool call still lacks a capability.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in allowed and name not in lambda_names:
                raise ValueError(f"undeclared policy capability: {name}")
        for child in ast.iter_child_nodes(node):
            check_names(child, lambda_names)
    check_names(tree)
    if stack:
        entries = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_task"]
        if len(entries) != 1:
            raise ValueError("stack.py must define run_task(task)")
        args = entries[0].args
        if (len(args.posonlyargs + args.args) != 1 or args.vararg or args.kwarg or args.kwonlyargs or args.defaults):
            raise ValueError("run_task must accept exactly one task argument")
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)) and not (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
                raise ValueError("stack.py contains a top-level action or constant")
        import re
        pattern = (re.compile(r"\b(?:" + "|".join(re.escape(x.lower()) for x in forbidden_literals if x) + r")\b")
                   if any(forbidden_literals) else None)
        diagnostic_literals = _diagnostic_literals(tree)
        for node in ast.walk(tree):
            if (pattern and isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node not in diagnostic_literals and pattern.search(node.value.lower())):
                raise ValueError(f"Recursia stack contains an object/colour/task literal at line {node.lineno}; use task data")
        # Definitions are imported before run_task; defaults/annotations must
        # not invoke tools as an unvalidated module-initialization side effect.
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                evaluated = [*node.decorator_list, *node.args.defaults,
                             *(value for value in node.args.kw_defaults if value is not None)]
                evaluated += [a.annotation for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                              if a.annotation is not None]
                if node.returns is not None:
                    evaluated.append(node.returns)
                if any(isinstance(part, ast.Call) for value in evaluated for part in ast.walk(value)):
                    raise ValueError("stack definition must not call functions at import time")
    return tree


def make_namespace(task, tools):
    if set(tools) != set(TOOLS):
        raise ValueError("runtime must expose exactly the six declared primitives")
    safe = {name: getattr(builtins, name) for name in BUILTINS if name != "globals"}
    safe.update(print=print, __import__=_policy_import)
    namespace = {"__builtins__": safe, "np": np, "Rotation": Rotation, "math": math,
                 "TASK": task, **tools}
    # Inspect only this episode's namespace. Do not expose host globals,
    # implementation builtins, or a dictionary that can replace tool bindings.
    safe["globals"] = lambda: MappingProxyType({name: value for name, value in namespace.items()
                                               if not name.startswith("__")})
    return namespace


def skill_names(code):
    if not code.strip():
        return set()
    tree = validate_policy(code)
    if any(not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)) for node in tree.body):
        raise ValueError("skill library may contain only functions and imports")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            evaluated = [*node.decorator_list, *node.args.defaults,
                         *(value for value in node.args.kw_defaults if value is not None)]
            evaluated += [arg.annotation for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                          if arg.annotation is not None]
            if node.returns is not None:
                evaluated.append(node.returns)
            if any(isinstance(part, ast.Call) for value in evaluated for part in ast.walk(value)):
                raise ValueError("skill definition must not call functions at injection time")
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
