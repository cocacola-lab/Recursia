# SPDX-FileCopyrightText: Copyright (c) 2026 Max Fu
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FR3 port of sim/cap/envs/trial.py's code-block loop.

Source: _run_single_trial, _annotate_code_blocks and launch_utils parsers.
The environment/model/video integrations are supplied by the real adapter.
Executed blocks are never replayed during regeneration. Python globals survive
steps, exactly as in CodeExecutionEnvBase. MULTITURN_LIMIT retains the simulation
bound (indices 0..10, at most eleven executed blocks).
"""
from __future__ import annotations
from typing import Any

MULTITURN_LIMIT = 10

def _extract_code(content: str) -> list[str]:
    """Extract Python code from Markdown fenced code block.

    Args:
        content: Raw model response

    Returns:
        Extracted Python code list
    """
    fence_start = "```python\n"
    fence_end = "```"
    start_idx = 0
    end_idx = len(content) + 1
    if fence_start in content:
        start_idx = content.find(fence_start) + len(fence_start)
        content = content[start_idx:]
    if fence_end in content:
        end_idx = content.rfind(fence_end)
        content = content[:end_idx]

    content = content.strip()
    # NOTE: might gen empty code block at the end
    # content_list = content.split("breakpoint_code_block()")

    return [content]


def _parse_multi_turn_decision(content: str) -> tuple[str, str | None]:
    """Parses the multi-turn decision response from the model

    Args:
        content: The model's response to the multi-turn decision prompt

    Returns:
        A tuple containing the decision and the new code (if any)
    """
    if content is not None and ("REGENERATE" in content):
        return "regenerate", content.split("REGENERATE")[1].strip()  # new code
    else:
        return "finish", content


def _annotate_code_blocks(
    code_blocks: list[str],
    code_block_metadata: list[dict[str, Any]],
) -> str:
    """Join code blocks into a single string with ``# Code block N`` headers."""
    annotated = []
    for i, (block, metadata) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
        annotated.append(f"# Code block {i}\n{block}")
    return "\n\n".join(annotated)



def run_trial(env, raw_code, decide, *, limit=MULTITURN_LIMIT, on_turn=None):
    """Execute -> observe -> REGENERATE/FINISH, using the simulation ordering."""
    code_blocks = _extract_code(raw_code)
    code_block_metadata = [{"generation": 0, "regenerated": False}] * len(code_blocks)
    all_responses = [{"decision": "initial", "block_idx": [0], "code_blocks": code_blocks.copy()}]
    code_block_idx = num_regenerations = num_finishes = 0
    turns = []
    info_step = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    model_error = None
    forced_finish = False
    obs = None
    while code_block_idx < len(code_blocks) and code_block_idx <= limit:
        code = code_blocks[code_block_idx]
        code_block_idx += 1
        obs, reward, terminated, truncated, info_step = env.step(code)
        turn = {"index": code_block_idx, "code": code, "info_step": info_step,
                "reward": reward, "terminated": terminated, "truncated": truncated}
        turns.append(turn)
        if on_turn:
            on_turn(turn, None)
        # Real controller/camera failures cannot be repaired by more motion.
        if terminated or truncated or "terminated episode" in info_step["stderr"]:
            if info_step.get("stop_reason") == "hardware_fault":
                forced_finish = True
                num_finishes += 1
                reasoning = "Hardware fault: finish motion immediately; source learning is read-only."
                turn.update(decision="finish", decision_origin="runtime", reasoning=reasoning)
                all_responses.append({"decision": "finish", "origin": "runtime",
                                      "stop_reason": "hardware_fault", "reasoning": reasoning})
                if on_turn:
                    on_turn(turn, all_responses)
            break
        try:
            decision, new_code, reasoning, prompt = decide(
                obs, code_blocks[:code_block_idx], info_step, turns)
            turn.update(decision=decision, reasoning=reasoning, decision_prompt=prompt)
            if decision == "regenerate":
                new_blocks = _extract_code(new_code)
                all_responses.append({"decision": "regenerate", "block_idx": [code_block_idx],
                                      "code_blocks": new_blocks, "reasoning": reasoning})
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
                code_blocks.extend(new_blocks)
                code_block_metadata.extend(
                    [{"generation": num_regenerations + 1, "regenerated": True,
                      "regenerated_at_idx": code_block_idx}] * len(new_blocks))
                num_regenerations += 1
            elif decision == "finish":
                all_responses.append({"decision": "finish", "reasoning": reasoning})
                num_finishes += 1
            else:
                raise ValueError(f"unexpected multi-turn decision: {decision}")
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {exc}"
            turn["model_error"] = model_error
        if on_turn:
            on_turn(turn, all_responses)
        if model_error or num_finishes:
            break
    if not num_finishes and code_block_idx < len(code_blocks):
        truncated = True
    executed_code = "\n\n".join(t["code"] for t in turns)
    return {"final_code": _annotate_code_blocks(code_blocks, code_block_metadata),
            "executed_code": executed_code, "code_blocks": code_blocks,
            "code_block_metadata": code_block_metadata, "turns": turns,
            "all_responses": all_responses, "info_step": info_step,
            "reward": reward, "terminated": terminated, "truncated": truncated,
            "num_regenerations": num_regenerations, "num_finishes": num_finishes,
            "forced_finish": forced_finish, "stop_reason": info_step.get("stop_reason"),
            "num_code_blocks": code_block_idx, "model_error": model_error}
