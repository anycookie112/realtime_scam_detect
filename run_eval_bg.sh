#!/usr/bin/env bash
# Background eval launcher with auto-resume on crash.
#
# Differences from run_eval.sh:
#   - Detached via setsid+nohup (survives ssh drops; no tmux dependency)
#   - Wraps the eval in a supervisor loop that re-launches with --resume
#     on any non-zero exit / SIGKILL, up to MAX_RETRIES times
#   - Logs to logs/eval_runs/<profile>_bg_<timestamp>.log
#
# Usage:
#   ./run_eval_bg.sh <profile>     # launch profile, auto-resume on crash
#   ./run_eval_bg.sh status        # show what's running
#   ./run_eval_bg.sh kill          # stop any running bg eval
#   ./run_eval_bg.sh tail          # follow latest log

set -euo pipefail
cd "$(dirname "$0")"

CONFIG_FILE="config/eval.yaml"
PIDFILE="logs/eval_runs/.bg_eval.pid"
MAX_RETRIES=10

mkdir -p logs/eval_runs

read_yaml() { uv run python3 -c "
import yaml
data = yaml.safe_load(open('$CONFIG_FILE'))
$1
" 2>/dev/null; }

action="${1:-}"

case "$action" in
    ""|help|-h|--help)
        echo "Usage: $0 <profile|command>"
        echo
        echo "Commands:"
        echo "  status   Show whether a bg eval is running"
        echo "  kill     Stop the running bg eval"
        echo "  tail     Follow the latest log"
        echo
        echo "Profiles (from $CONFIG_FILE):"
        read_yaml "
for name, prof in data['profiles'].items():
    print(f'  {name:<20} {prof.get(\"description\", \"\")}')"
        exit 0
        ;;
    status)
        if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "Running: PID $(cat $PIDFILE)"
            ps -fp "$(cat $PIDFILE)" 2>/dev/null
        else
            echo "Not running"
            [[ -f "$PIDFILE" ]] && rm "$PIDFILE"
        fi
        exit 0
        ;;
    kill)
        if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            pid="$(cat "$PIDFILE")"
            # Kill the supervisor + its whole process group so children die too
            pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
            kill -TERM -"$pgid" 2>/dev/null || true
            sleep 2
            kill -KILL -"$pgid" 2>/dev/null || true
            rm -f "$PIDFILE"
            echo "Killed bg eval (PID $pid, PGID $pgid)"
        else
            echo "No bg eval running"
            [[ -f "$PIDFILE" ]] && rm "$PIDFILE"
        fi
        exit 0
        ;;
    tail)
        latest=$(ls -t logs/eval_runs/*_bg_*.log 2>/dev/null | head -1)
        [[ -z "$latest" ]] && { echo "No bg log found"; exit 1; }
        echo "Tailing $latest (Ctrl+C to stop)"
        exec tail -f "$latest"
        ;;
esac

# Verify profile
prof_ok=$(read_yaml "print('yes' if '$action' in data['profiles'] else 'no')")
[[ "$prof_ok" != "yes" ]] && { echo "!! Unknown profile: $action"; exit 1; }

# Refuse if one is already running
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "!! Already running (PID $(cat $PIDFILE)). Use '$0 kill' first."
    exit 1
fi

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG_DIR="logs/eval_runs"
LOG_FILE="$LOG_DIR/${action}_bg_${TIMESTAMP}.log"
SCRIPT="$LOG_DIR/${action}_bg_${TIMESTAMP}.sh"

# Build the eval command from the profile
EVAL_CMD=$(read_yaml "
prof = data['profiles']['$action']
script = data['defaults']['script']
args = ' '.join(prof.get('args', []))
if 'sequential_configs' in prof:
    cfgs = prof['sequential_configs']
    print(' && '.join(f'uv run {script} --only={c} {args}' for c in cfgs))
else:
    print(f'uv run {script} {args}')
")

# Supervisor script — runs the eval, retries with --resume on crash
cat > "$SCRIPT" <<EOF
#!/usr/bin/env bash
cd "$(pwd)"
echo "Eval supervisor started at \$(date)"
echo "Profile: $action"
echo "Cmd:     $EVAL_CMD"
echo "Max retries: $MAX_RETRIES"
echo

attempt=1
cmd="$EVAL_CMD"
while (( attempt <= $MAX_RETRIES )); do
    echo
    echo "═══ Attempt \$attempt/$MAX_RETRIES — \$(date) ═══"
    echo "Running: \$cmd"
    set +e
    eval \$cmd
    rc=\$?
    set -e
    echo
    echo "Attempt \$attempt exited with code \$rc at \$(date)"
    if [[ \$rc -eq 0 ]]; then
        echo "✓ Eval completed successfully"
        break
    fi
    echo "✗ Eval crashed/killed. Retrying with --resume in 30s..."
    sleep 30
    # Append --resume for subsequent attempts (idempotent if already there)
    if [[ "\$cmd" != *"--resume"* ]]; then
        cmd="\$cmd --resume"
    fi
    attempt=\$((attempt + 1))
done

if (( attempt > $MAX_RETRIES )); then
    echo "!! Hit max retries ($MAX_RETRIES). Giving up."
fi
echo "Supervisor done at \$(date)"
EOF
chmod +x "$SCRIPT"

# Launch detached: setsid creates a new process group + session,
# nohup ignores SIGHUP, &disown removes from shell job table.
setsid nohup bash "$SCRIPT" > "$LOG_FILE" 2>&1 < /dev/null &
SUPERVISOR_PID=$!
disown
echo "$SUPERVISOR_PID" > "$PIDFILE"

echo "✓ Eval launched in background"
echo "  Profile: $action"
echo "  PID:     $SUPERVISOR_PID"
echo "  Log:     $LOG_FILE"
echo "  Status:  $0 status"
echo "  Tail:    $0 tail"
echo "  Kill:    $0 kill"
echo
echo "Safe to close this terminal — the eval keeps running."
