#!/usr/bin/env bash
# Run eval profiles defined in config/eval.yaml under tmux.
#
# Usage:
#   ./run_eval.sh <profile>      # launch a profile in detached tmux
#   ./run_eval.sh list           # show available profiles
#   ./run_eval.sh attach         # reattach to running session
#   ./run_eval.sh kill           # stop running eval
#   ./run_eval.sh sessions       # list active tmux sessions
#
# Profiles are defined in config/eval.yaml — edit that file to add new ones.

set -euo pipefail

cd "$(dirname "$0")"

CONFIG_FILE="config/eval.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "!! Config file not found: $CONFIG_FILE"
    exit 1
fi

# Read defaults from YAML (uses uv to access PyYAML from .venv)
read_yaml() {
    uv run python3 -c "
import yaml, sys
data = yaml.safe_load(open('$CONFIG_FILE'))
$1
" 2>/dev/null
}

SESSION=$(read_yaml "print(data['defaults']['session'])")
LOG_DIR=$(read_yaml "print(data['defaults']['log_dir'])")
SCRIPT=$(read_yaml "print(data['defaults']['script'])")

mkdir -p "$LOG_DIR"

# ── Action dispatch ─────────────────────────────────────────────────────────

action="${1:-}"
RESUME_FLAG=""

# `resume <profile>` shortcut — same as the profile but with --resume appended,
# which makes the eval pick up where the latest run died.
if [[ "$action" == "resume" ]]; then
    if [[ -z "${2:-}" ]]; then
        echo "Usage: $0 resume <profile>"
        echo "       (resumes the latest logs/eval/<timestamp>/ run)"
        exit 1
    fi
    action="$2"
    RESUME_FLAG=" --resume"
fi

case "$action" in
    ""|help|-h|--help)
        echo "Usage: $0 <profile|command>"
        echo
        echo "Commands:"
        echo "  list              Show available eval profiles"
        echo "  attach            Reattach to running tmux session"
        echo "  kill              Stop the running eval"
        echo "  sessions          List active tmux sessions"
        echo "  resume <profile>  Re-run a profile, skipping clips already in the latest checkpoint"
        echo
        echo "Profiles (from $CONFIG_FILE):"
        read_yaml "
for name, prof in data['profiles'].items():
    desc = prof.get('description', '')
    print(f'  {name:<20} {desc}')
"
        exit 0
        ;;

    list|ls|profiles)
        echo "Profiles defined in $CONFIG_FILE:"
        echo
        read_yaml "
for name, prof in data['profiles'].items():
    desc = prof.get('description', '(no description)')
    print(f'  \033[1;36m{name}\033[0m')
    print(f'    {desc}')
    args = prof.get('args', [])
    if 'sequential_configs' in prof:
        print(f'    \033[2msequential configs: {prof[\"sequential_configs\"]}\033[0m')
    print(f'    \033[2margs: {\" \".join(args)}\033[0m')
    print()
"
        exit 0
        ;;

    attach)
        tmux attach -t "$SESSION"
        exit 0
        ;;

    kill)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            tmux kill-session -t "$SESSION"
            echo "Killed session $SESSION"
        else
            echo "No session running"
        fi
        exit 0
        ;;

    sessions)
        tmux ls 2>/dev/null || echo "No tmux sessions"
        exit 0
        ;;
esac

# ── Profile execution ───────────────────────────────────────────────────────

# Verify the profile exists in the YAML
profile_exists=$(read_yaml "print('yes' if '$action' in data['profiles'] else 'no')")
if [[ "$profile_exists" != "yes" ]]; then
    echo "!! Unknown profile: $action"
    echo
    echo "Available profiles:"
    read_yaml "
for name in data['profiles']:
    print(f'  - {name}')
"
    exit 1
fi

# Refuse to start if a session is already running
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "!! Session '$SESSION' already running."
    echo "   Attach: $0 attach"
    echo "   Kill:   $0 kill"
    exit 1
fi

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
TAG=""
[[ -n "$RESUME_FLAG" ]] && TAG="_resume"
LOG_FILE="$LOG_DIR/${action}${TAG}_${TIMESTAMP}.log"
RUN_SCRIPT="$LOG_DIR/${action}${TAG}_${TIMESTAMP}.sh"

# Build the eval shell script from the YAML profile and write it to a file.
# Inlining via bash -c "..." breaks on quoting; a temp script avoids escape hell.
read_yaml "
prof = data['profiles']['$action']
args = prof.get('args', [])
script = data['defaults']['script']
arg_str = ' '.join(args)

print('#!/usr/bin/env bash')
print('set -uo pipefail')
print('cd \"\$(dirname \"\$0\")/../..\"')   # back to project root
print()
print('echo \"Eval started at \$(date)\"')
print(f'echo \"Profile: $action\"')
print(f'echo \"Log:     $LOG_FILE\"')
print('echo \"PID:     \$\$\"')
print('echo')

if 'sequential_configs' in prof:
    cfgs = prof['sequential_configs']
    print(f'echo \"=== Profile: $action (sequential mode)$RESUME_FLAG ===\"')
    print('date')
    for cfg in cfgs:
        print('echo')
        print(f'echo \"═══ Config: {cfg} ═══\"')
        print(f'time uv run {script} --only={cfg} {arg_str}$RESUME_FLAG || echo \"  ! {cfg} failed, continuing\"')
    print('echo')
    print('echo \"=== ALL DONE ===\"')
    print('date')
else:
    print(f'echo \"=== Profile: $action$RESUME_FLAG ===\"')
    print(f'echo \"Args: {arg_str}$RESUME_FLAG\"')
    print('date')
    print(f'time uv run {script} {arg_str}$RESUME_FLAG')
    print('echo')
    print('echo \"=== DONE ===\"')
    print('date')

print()
print('echo')
print('echo \"────────────────────────────────────────────────────────\"')
print('echo \"Eval complete. Press any key to close this tmux pane\"')
print('echo \"(or detach with Ctrl+B then D to keep it).\"')
print('read -n 1')
" > "$RUN_SCRIPT"

chmod +x "$RUN_SCRIPT"

# Launch the script in tmux, with output tee'd to the log file.
tmux new-session -d -s "$SESSION" \
    "bash '$RUN_SCRIPT' 2>&1 | tee '$LOG_FILE'; echo; echo 'Press any key to close.'; read -n 1"

echo "✓ Eval started in tmux session '$SESSION'"
echo "  Profile: $action"
echo "  Log:     $LOG_FILE"
echo
echo "  Attach:  $0 attach"
echo "  Kill:    $0 kill"
echo "  Detach:  Ctrl+B then D (once attached)"
echo
echo "Reports will be written to logs/eval/<timestamp>/"
