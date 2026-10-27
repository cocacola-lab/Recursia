"""Orchestration program boundary v2: AST validation + a restricted INTERPRETER. Generated code is never exec'd or eval'd.

Adopted from the N11-client reference design (rsi_orchestration/program.py): the program is a single `def run(context, stack)` whose
statements and expressions come from a small allow-list and are interpreted node by node, so no Python object, attribute, builtin
or module is reachable from generated code. Additions over that design: iteration over held list values (lists cannot grow: there
is no attribute call, so no append), `len` / `str`, `+` on two strings / two bounded lists / two integers, `continue`, and the v1 static capability check
(relation / side / object_side / vocabulary / reference kind) on every literal `stack.subgoal(...)` before any motion.
Budgets: AST nodes, interpreted steps, stack calls. Not a hardened OS sandbox (the interpreter runs in the worker process)."""
from __future__ import annotations

import ast

CHECKER_VERSION = "orch_check_v2_interpreter_20261008"
STACK_METHODS = ("vocabulary", "capabilities", "observe", "robot_state", "subgoal", "check", "plan", "reference", "locate",
                 "execute", "recover", "log", "finish", "budget")
FUNCS = {"len": len, "str": lambda x: str(x)[:500]}
MAX_NODES, MAX_STEPS, MAX_CALLS, MAX_RANGE, MAX_ITER, MAX_CONTAINER, MAX_STR, MAX_DEPTH = 3000, 5000, 80, 8, 16, 32, 2000, 8
RESERVED = ("context", "stack", "range", "len", "str")


class ProgramRejected(ValueError):
    pass


class ProgramBudgetExceeded(BaseException):
    """BaseException: nothing in the stack wrapper may mask it; handled by execute_program."""


class _Return(Exception):
    def __init__(self, value):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


def _literal(node):
    try:
        return True, ast.literal_eval(node)
    except Exception:  # noqa: BLE001
        return False, None


class Validator:
    """Allow-list walk; collects errors (all of them, for the report) and literal stack.subgoal calls."""

    def __init__(self):
        self.errors = []; self.subgoal_calls = []; self.recover_actions = []

    def err(self, node, kind, msg):
        self.errors.append({"kind": kind, "line": getattr(node, "lineno", None), "msg": msg})

    def name(self, node, ident):
        if not isinstance(ident, str) or ident.startswith("_") or "__" in ident:
            self.err(node, "private_name", f"name {ident!r} is not allowed")

    def expr(self, n):
        if isinstance(n, ast.Constant):
            v = n.value
            if type(v) not in (str, int, float, bool, type(None)) or (isinstance(v, str) and len(v) > MAX_STR) or (type(v) is int and abs(v) > 10 ** 6):
                self.err(n, "constant", "only short str / int / float / bool / None literals")
        elif isinstance(n, ast.Name):
            self.name(n, n.id)
            if n.id == "stack":
                self.err(n, "stack_reference_escape", "`stack` may only be used as `stack.<method>(...)`")
        elif isinstance(n, (ast.List, ast.Tuple)):
            if len(n.elts) > MAX_CONTAINER:
                self.err(n, "container_size", f"at most {MAX_CONTAINER} elements")
            for x in n.elts:
                if isinstance(x, ast.Starred):
                    self.err(x, "forbidden_construct", "starred expressions are not allowed")
                else:
                    self.expr(x)
        elif isinstance(n, ast.Dict):
            if len(n.keys) > MAX_CONTAINER:
                self.err(n, "container_size", f"at most {MAX_CONTAINER} keys")
            for k, v in zip(n.keys, n.values):
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str)) or k.value.startswith("_"):
                    self.err(n, "dictionary_key", "dict keys must be string literals not starting with _")
                self.expr(v)
        elif isinstance(n, ast.Subscript):
            if isinstance(n.slice, ast.Slice):
                self.err(n, "forbidden_construct", "slices are not allowed")
            else:
                self.expr(n.value); self.expr(n.slice)
        elif isinstance(n, ast.BoolOp):
            for x in n.values:
                self.expr(x)
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            self.expr(n.operand)
        elif isinstance(n, ast.Compare):
            self.expr(n.left)
            for x in n.comparators:
                self.expr(x)
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            self.expr(n.left); self.expr(n.right)   # runtime: str+str, list+list (bounded), int+int
        elif isinstance(n, ast.IfExp):
            self.expr(n.test); self.expr(n.body); self.expr(n.orelse)
        elif isinstance(n, ast.Call):
            if any(isinstance(a, ast.Starred) for a in n.args) or any(k.arg is None for k in n.keywords):
                self.err(n, "forbidden_construct", "*args / **kwargs are not allowed")
            f = n.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "stack":
                if f.attr not in STACK_METHODS:
                    self.err(n, "stack_method", f"stack.{f.attr} is not part of the interface")
                if f.attr == "subgoal":
                    self.subgoal_calls.append(n)
                if f.attr == "recover":
                    a = n.args[1] if len(n.args) > 1 else next((k.value for k in n.keywords if k.arg == "action"), None)
                    ok, v = _literal(a) if a is not None else (False, None)
                    if ok:
                        self.recover_actions.append((n, v))
            elif isinstance(f, ast.Name) and f.id in FUNCS:
                if len(n.args) != 1 or n.keywords:
                    self.err(n, "call_signature", f"{f.id}() takes exactly one argument")
            elif isinstance(f, ast.Attribute):
                self.err(n, "attribute", f"attribute call .{f.attr}() is not allowed (use result['key'] and `in`)")
                if not (isinstance(f.value, ast.Name) and f.value.id == "stack"):
                    self.expr(f.value)   # report what the attribute is read from, too
            else:
                self.err(n, "unauthorized_call", "only stack.<method>(...), len(x) and str(x) may be called")
                if not isinstance(f, ast.Name):
                    self.expr(f)
            for a in n.args:
                if not isinstance(a, ast.Starred):
                    self.expr(a)
            for k in n.keywords:
                if k.arg is not None:
                    self.name(k, k.arg); self.expr(k.value)
        elif isinstance(n, ast.Attribute):
            self.err(n, "attribute", f"attribute access .{n.attr} is not allowed (use result['key'])")
        elif isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            self.err(n, "unbounded_loop", "comprehensions are not allowed")
        elif isinstance(n, ast.Lambda):
            self.err(n, "forbidden_construct", "lambda is not allowed")
        else:
            self.err(n, "unsupported_expression", f"{type(n).__name__} is not allowed")

    def block(self, body, depth, loops):
        if depth > MAX_DEPTH:
            self.err(body[0] if body else None, "nesting_limit", f"nesting deeper than {MAX_DEPTH}")
            return
        for n in body:
            if isinstance(n, ast.Assign):
                if len(n.targets) != 1 or not isinstance(n.targets[0], ast.Name):
                    self.err(n, "assignment_target", "only `name = value` assignments")
                elif n.targets[0].id in RESERVED:
                    self.err(n, "stack_modification", f"cannot rebind {n.targets[0].id!r}")
                else:
                    self.name(n, n.targets[0].id)
                self.expr(n.value)
            elif isinstance(n, ast.Expr):
                self.expr(n.value)
            elif isinstance(n, ast.Return):
                if n.value is not None:
                    self.expr(n.value)
            elif isinstance(n, ast.If):
                self.expr(n.test); self.block(n.body, depth + 1, loops); self.block(n.orelse, depth + 1, loops)
            elif isinstance(n, ast.For):
                if not isinstance(n.target, ast.Name) or n.target.id in RESERVED:
                    self.err(n, "loop_target", "loop target must be a plain name")
                else:
                    self.name(n, n.target.id)
                it = n.iter
                if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
                    if not (len(it.args) == 1 and not it.keywords and isinstance(it.args[0], ast.Constant) and type(it.args[0].value) is int and 0 <= it.args[0].value <= MAX_RANGE):
                        self.err(n, "unbounded_loop", f"range() takes one integer literal 0..{MAX_RANGE}")
                elif isinstance(it, (ast.Name, ast.List, ast.Tuple, ast.Subscript)):
                    self.expr(it)   # runtime: a list / tuple of at most MAX_ITER items (lists cannot grow: no attribute calls)
                else:
                    self.err(n, "loop_iterable", "for-loops iterate over range(<literal>) or a held list")
                self.block(n.body, depth + 1, loops + 1); self.block(n.orelse, depth + 1, loops)
            elif isinstance(n, (ast.Break, ast.Continue)):
                if loops == 0:
                    self.err(n, "break_outside_loop", "break / continue outside a loop")
            elif isinstance(n, ast.Pass):
                pass
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                self.err(n, "forbidden_construct", "imports are not allowed")
            elif isinstance(n, ast.While):
                self.err(n, "forbidden_construct", "while is not allowed")
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                self.err(n, "nested_def", "helper functions / classes are not allowed")
            elif isinstance(n, (ast.Try, ast.Raise, ast.With, ast.Global, ast.Nonlocal, ast.Delete, ast.AugAssign, ast.AnnAssign)):
                self.err(n, "forbidden_construct", f"{type(n).__name__} is not allowed")
            else:
                self.err(n, "unsupported_statement", f"{type(n).__name__} is not allowed")


def parse_and_validate(src):
    v = Validator()
    if not isinstance(src, str) or len(src.encode()) > 65536:
        v.err(None, "size", "program larger than 64 KiB"); return None, v
    try:
        tree = ast.parse(src, filename="<orch_program>")
    except (SyntaxError, RecursionError) as e:
        v.err(None, "syntax", str(e)); return None, v
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        v.err(None, "ast_size", f"more than {MAX_NODES} AST nodes"); return None, v
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef):
        v.err(body[0] if body else None, "top_level", "the program must be exactly one `def run(context, stack):` (no imports, helpers or top-level code)")
        return None, v
    f = body[0]; a = f.args
    if (f.name != "run" or [x.arg for x in a.args] != ["context", "stack"] or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or a.defaults
            or f.decorator_list or f.returns or any(x.annotation for x in a.args)):
        v.err(f, "entry", "the entry point must be exactly `def run(context, stack):`")
    v.block(f.body, 0, 0)
    return f, v


def check_program(src, vocab=None, supported=None, object_sides=None, part_relations=(), recovery_actions=None):
    """-> report {ok, errors[], static_subgoals[], checker_version}. Structure first, then static capability on literal subgoals."""
    f, v = parse_and_validate(src)
    static = [_static_subgoal(c, v.errors, vocab, supported, object_sides, part_relations) for c in v.subgoal_calls]
    for node, act in v.recover_actions:
        if recovery_actions is not None and act not in recovery_actions:
            v.err(node, "capability_recovery", f"recovery action {act!r} is not supported (supported: {list(recovery_actions)})")
    return {"checker_version": CHECKER_VERSION, "ok": f is not None and not v.errors, "errors": v.errors, "static_subgoals": static,
            "n_lines": (src.count("\n") + 1) if isinstance(src, str) else 0}


def _static_subgoal(call, errors, vocab, supported, object_sides, part_relations):
    """Literal arguments of stack.subgoal(...) checked against the frozen capability tables (la_dispatch's checks + the contract)."""
    def err(kind, msg):
        errors.append({"kind": kind, "line": call.lineno, "msg": msg})
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    names = ("relation", "object", "reference", "part", "side", "object_side", "instance")
    for i, a in enumerate(call.args[:len(names)]):
        kw.setdefault(names[i], a)
    rec = {"line": call.lineno}
    for k in names:
        if k in kw:
            ok, val = _literal(kw[k]); rec[k] = val if ok else "<runtime>"
    if vocab is None or supported is None:
        return rec
    rel = rec.get("relation")
    if rel is None:
        err("capability", "stack.subgoal needs a relation"); return rec
    sw = vocab.get("side_words") or {}
    if rel != "<runtime>":
        if rel not in supported:
            err("capability_relation", f"relation {rel!r} is not implemented by the frozen stack (supported: {sorted(supported)})"); return rec
        side = rec.get("side")
        if side not in (None, "<runtime>"):
            if sw.get(side, side) not in supported[rel]:
                err("capability_side", f"side {side!r} is not implemented for relation {rel!r} (implemented codes: {list(supported[rel])})")
        elif side is None and None not in supported[rel] and not _phrase_has_side(rec, sw):
            err("capability_side", f"relation {rel!r} needs a side word (implemented codes: {list(supported[rel])})")
        osd = rec.get("object_side")
        if osd not in (None, "<runtime>") and sw.get(osd, osd) not in (object_sides or ()):
            err("capability_object_side", f"object_side {osd!r} is not implemented")
        if rel not in tuple(part_relations or ()):
            ref = rec.get("reference")
            if ref not in (None, "<runtime>") and ref in (vocab.get("articulated") or ()) and ref not in (vocab.get("supports") or ()) and ref not in (vocab.get("objects") or ()):
                err("capability_reference_kind", f"relation {rel!r} needs a support or object reference; {ref!r} is an articulated part (not implemented as a container)")
            if rec.get("part") is not None and rel != "in_sliding":
                err("capability_part_not_allowed", f"relation {rel!r} takes no part argument")
    for k, tables in (("object", ("objects",)), ("reference", ("supports", "objects", "articulated")), ("part", ("articulated",))):
        val = rec.get(k)
        if val not in (None, "<runtime>") and not any(val in (vocab.get(t) or ()) for t in tables):
            err("capability_vocabulary", f"{k} phrase {val!r} is not in the frozen vocabulary for this instruction")
    return rec


def _phrase_has_side(rec, sw):
    """A reference phrase carrying its own side word ("left plate") supplies the side (la_subgoal rule (a))."""
    ref = rec.get("reference")
    return isinstance(ref, str) and len([t for t in ref.split() if t in sw]) == 1


class Interpreter:
    """Interprets a validated `run` FunctionDef node by node. The only effects are stack.<method> calls."""

    def __init__(self, fn, stack, max_steps=MAX_STEPS, max_calls=MAX_CALLS, trace=None):
        self.fn, self.stack, self.max_steps, self.max_calls = fn, stack, max_steps, max_calls
        self.steps = 0; self.calls = 0; self.trace = trace or (lambda *a, **k: None); self.vars = {}

    def run(self, context):
        self.vars = {"context": context}
        try:
            self.block(self.fn.body)
        except _Return as r:
            return r.value
        return None

    def tick(self):
        self.steps += 1
        if self.steps > self.max_steps:
            raise ProgramBudgetExceeded(f"interpreted-step budget {self.max_steps} exhausted")

    def value(self, n):
        self.tick()
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in self.vars:
                raise NameError(f"name {n.id!r} is not defined")
            return self.vars[n.id]
        if isinstance(n, ast.List):
            return [self.value(x) for x in n.elts]
        if isinstance(n, ast.Tuple):
            return tuple(self.value(x) for x in n.elts)
        if isinstance(n, ast.Dict):
            return {k.value: self.value(v) for k, v in zip(n.keys, n.values)}
        if isinstance(n, ast.Subscript):
            obj, key = self.value(n.value), self.value(n.slice)
            if type(key) not in (str, int) or (isinstance(key, str) and key.startswith("_")):
                raise TypeError("index must be an int or a public str key")
            if not isinstance(obj, (dict, list, tuple, str)):
                raise TypeError(f"cannot index a {type(obj).__name__}")
            return obj[key]
        if isinstance(n, ast.BoolOp):
            r = self.value(n.values[0])
            for x in n.values[1:]:
                if isinstance(n.op, ast.And) and not r:
                    return r
                if isinstance(n.op, ast.Or) and r:
                    return r
                r = self.value(x)
            return r
        if isinstance(n, ast.UnaryOp):
            return not self.value(n.operand)
        if isinstance(n, ast.IfExp):
            return self.value(n.body) if self.value(n.test) else self.value(n.orelse)
        if isinstance(n, ast.BinOp):
            a, b = self.value(n.left), self.value(n.right)
            if isinstance(a, str) and isinstance(b, str) and len(a) + len(b) <= 4 * MAX_STR:
                return a + b
            if isinstance(a, list) and isinstance(b, list) and len(a) + len(b) <= MAX_CONTAINER:
                return a + b
            if type(a) is int and type(b) is int and abs(a + b) <= 10 ** 6:
                return a + b
            raise TypeError("+ joins two strings, two lists (<= %d items) or two integers" % MAX_CONTAINER)
        if isinstance(n, ast.Compare):
            a = self.value(n.left)
            for op, x in zip(n.ops, n.comparators):
                b = self.value(x)
                ok = {ast.Eq: lambda: a == b, ast.NotEq: lambda: a != b, ast.In: lambda: a in b, ast.NotIn: lambda: a not in b,
                      ast.Is: lambda: a is b, ast.IsNot: lambda: a is not b, ast.Lt: lambda: a < b, ast.LtE: lambda: a <= b,
                      ast.Gt: lambda: a > b, ast.GtE: lambda: a >= b}[type(op)]()
                if not ok:
                    return False
                a = b
            return True
        if isinstance(n, ast.Call):
            args = [self.value(x) for x in n.args]; kwargs = {k.arg: self.value(k.value) for k in n.keywords}
            if isinstance(n.func, ast.Name):
                return FUNCS[n.func.id](*args)
            self.calls += 1
            if self.calls > self.max_calls:
                raise ProgramBudgetExceeded(f"stack-call budget {self.max_calls} exhausted")
            return getattr(self.stack, n.func.attr)(*args, **kwargs)   # attr is in STACK_METHODS (validated)
        raise ProgramRejected(f"invalid expression {type(n).__name__}")

    def block(self, body):
        for n in body:
            self.tick()
            if isinstance(n, ast.Assign):
                self.vars[n.targets[0].id] = self.value(n.value)
            elif isinstance(n, ast.Expr):
                self.value(n.value)
            elif isinstance(n, ast.Return):
                raise _Return(self.value(n.value) if n.value is not None else None)
            elif isinstance(n, ast.If):
                taken = bool(self.value(n.test)); self.trace("branch", line=n.lineno, taken=taken)
                self.block(n.body if taken else n.orelse)
            elif isinstance(n, ast.For):
                it = n.iter
                if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
                    seq = list(range(it.args[0].value))
                else:
                    seq = self.value(it)
                    if not isinstance(seq, (list, tuple)) or len(seq) > MAX_ITER:
                        raise TypeError(f"for-loops iterate over a list of at most {MAX_ITER} items")
                    seq = list(seq)
                broke = False
                for item in seq:
                    self.tick(); self.vars[n.target.id] = item
                    try:
                        self.block(n.body)
                    except _Continue:
                        continue
                    except _Break:
                        broke = True; break
                if not broke:
                    self.block(n.orelse)
            elif isinstance(n, ast.Break):
                raise _Break()
            elif isinstance(n, ast.Continue):
                raise _Continue()
            elif isinstance(n, ast.Pass):
                pass
            else:
                raise ProgramRejected(f"invalid statement {type(n).__name__}")
