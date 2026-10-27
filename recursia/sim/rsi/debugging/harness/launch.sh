#!/usr/bin/env bash
# One-command start of an RSI zero-shot transfer debugging run (the human's only action).
#   bash harness/launch.sh <round_dir> <authorization.json>
# 1. preflight (perception 8114-8116, GPUs listed, no Long-Pro in the round);
# 2. rsi.py init (if the round does not exist yet);
# 3. VLM proxy on 127.0.0.1:8117 with this round's ledger and the authorized cap (detached process, pid in logs/proxy.pid; the key stays there);
# 4. the outer-loop agent (Opus) in tmux window "outer", prompt = outer-loop-agent-prompt.md with {R} filled.
# Extra Claude Code flags (e.g. a permission mode so the agent runs unattended) go in CLAUDE_ARGS.
set -euo pipefail
R=$(realpath -m "$1"); AUTH=$(realpath "$2")
HERE=$(cd "$(dirname "$0")" && pwd); RUNBOOK=$(dirname "$HERE")
ROOT=${RECURSIA_SIM_ROOT:-$(cd "$RUNBOOK/../.." && pwd)}; export RECURSIA_SIM_ROOT=$ROOT
SESSION=${RSI_TMUX_SESSION:-rsi-debug}
PY="$ROOT/.venv-libero/bin/python3"

for p in 8114 8115 8116; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$p/health" || echo 000)
  [[ "$code" != "000" ]] || { echo "perception server on port $p is down (see recursia/sim/README.md § Perception servers)"; exit 2; }
done
command -v claude >/dev/null || { echo "claude CLI not found"; exit 2; }

if [[ ! -f "$R/state.json" ]]; then
  "$PY" "$HERE/rsi.py" --round "$R" init --authorization "$AUTH"
fi
if grep -rIlE "libero_10|LIBERO-Long" "$R" --include='*.json' --include='*.txt' -q 2>/dev/null; then
  echo "Long-Pro content found in $R; refusing to start"; exit 2
fi

CAP=$("$PY" -c "import json;print(json.load(open('$R/authorization.json'))['vlm_usd_cap'])")
if ! curl -s --max-time 3 http://127.0.0.1:8117/health >/dev/null; then
  [[ -n "${ANTHROPIC_API_KEY:-}" ]] || { echo "start the proxy yourself or export ANTHROPIC_API_KEY for it"; exit 2; }
  # detached child of THIS shell, so it inherits the key (a tmux window would get the tmux server's old environment)
  ( cd "$ROOT" && setsid nohup .venv-vlmproxy/bin/python3 "$R/code/vlm_proxy.py" --port 8117 --cap-usd "$CAP" \
      --ledger "$R/logs/vlm_ledger.jsonl" > "$R/logs/proxy.out" 2>&1 & echo $! > "$R/logs/proxy.pid" )
  sleep 6
  env -u ANTHROPIC_API_KEY "$PY" "$R/code/vlm_probe.py" || { echo "VLM probe failed; see $R/logs/proxy.out"; exit 2; }
fi
curl -s --max-time 3 http://127.0.0.1:8117/health; echo

PROMPT_FILE="$R/logs/outer_loop_prompt.txt"
"$PY" - "$RUNBOOK/outer-loop-agent-prompt.md" "$R" > "$PROMPT_FILE" <<'EOF'
import re, sys
text = open(sys.argv[1]).read()
block = re.search(r"```\n(.*?)\n```", text, re.S).group(1)
print(block.replace("{R}", sys.argv[2]))
print(f"\nRunbook directory: {sys.argv[1].rsplit('/', 1)[0]}. Harness CLI: {sys.argv[2]}/code/rsi.py --round {sys.argv[2]} <command>.")
EOF
tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n shell
tmux new-window -t "$SESSION" -n outer \
  "cd '$ROOT' && env -u ANTHROPIC_API_KEY claude --model claude-opus-5-5 ${CLAUDE_ARGS:-} \"\$(cat '$PROMPT_FILE')\"; sleep infinity"
echo "LAUNCHED: tmux attach -t $SESSION (window 'outer'). Stop the run with: touch $R/STOP"
