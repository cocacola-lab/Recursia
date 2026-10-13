# SPDX-FileCopyrightText: Copyright (c) 2026 Max Fu
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent CodeExecutionEnvBase executor, with FR3 capability bindings."""
import contextlib
import io
import sys
import traceback
from typing import Any
from ..capabilities import make_namespace, validate_policy

class Tee(io.TextIOBase):
    """This allows streaming stdout and stderr to both the console and a buffer
    (enables breakpointing for debugging!)
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


class CodeExecutor:
    def __init__(self, task, tools, *, printer=None):
        self._exec_globals = make_namespace(task, tools)
        if printer is not None:
            self._exec_globals["__builtins__"]["print"] = printer

    def _exec_user_code(self, code: str) -> dict[str, Any]:
        stdout_buffer = io.StringIO()
        tee_out = Tee(sys.stdout, stdout_buffer)
        stderr_buffer = io.StringIO()
        tee_err = Tee(sys.stderr, stderr_buffer)
        ok = True
        execution_status = "validation_rejected"
        try:
            with (
                contextlib.redirect_stdout(tee_out),
                contextlib.redirect_stderr(tee_err),
            ):
                tree = validate_policy(code, known_names=self._exec_globals)
                compiled = compile(tree, "<aspire-block>", "exec")
                execution_status = "execution_error"
                exec(compiled, self._exec_globals, self._exec_globals)
                execution_status = "executed"
        except BaseException:  # defensive; propagate minimal info
            ok = False
            # Always print full traceback to the redirected stderr (tee -> console and buffer)
            traceback.print_exc(file=tee_err)

        return {
            "ok": ok,
            "execution_status": execution_status,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "result": self._exec_globals.get("RESULT"),
        }

    def run(self, code):
        return self._exec_user_code(code)
