#!/usr/bin/env bash
#
# results_watcher.sh -- self-contained "results watcher" that runs ON the Lambda
# box. It waits for the LoRA training run (frozen arm, then lora arm) to finish,
# pushes whatever result JSONs exist to a NEW GitHub branch, and -- only if the
# push succeeds -- self-terminates the box. If the push fails it leaves the box
# alive (loudly) so results can be rescued manually before the 8h watchdog kills
# it.
#
# Deploy (from a machine WITH network, e.g. your laptop):
#   scp scripts/results_watcher.sh ubuntu@<ip>:~/microbe-foundation/scripts/
#   ssh ubuntu@<ip> 'umask 077; cat > ~/.gh_token' <<< "$GH_TOKEN"
#   ssh ubuntu@<ip> 'cd microbe-foundation && nohup bash scripts/results_watcher.sh >>~/watcher.log 2>&1 & disown'
#
# Requirements on the box:
#   - run from inside the repo (cwd = ~/microbe-foundation)
#   - ~/.gh_token contains a fine-grained PAT (Contents: read/write), chmod 600
#   - ~/run_env.sh defines INSTANCE_ID and LAMBDA_API_KEY (used to terminate)
#
# Design notes:
#   - We authenticate to GitHub with a one-shot GIT_ASKPASS helper so the token
#     is NEVER written to .git/config and NEVER appears in process args / `ps`.
#   - We push to a fresh, uniquely-named branch (never the feature branch) so a
#     push can never be a non-fast-forward / divergence failure.
#
set -uo pipefail   # NOTE: intentionally NOT -e; we want to handle errors and
                   # keep the box alive on failure rather than dying silently.

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
REPO="miyu-horiuchi/microbe-foundation"
RESULTS_BRANCH="results/lora-8xa100-$(date -u +%Y%m%d-%H%M)"

TOKEN_FILE="$HOME/.gh_token"
LOG="$HOME/watcher.log"
RUN_ENV="$HOME/run_env.sh"

RESULTS_DIR="runs/lora"
FROZEN_JSON="$RESULTS_DIR/frozen_family_s0.json"
LORA_JSON="$RESULTS_DIR/lora_family_s0.json"

POLL_SECS=120          # poll cadence
EMPTY_LIMIT=3          # consecutive "no training procs" checks before declaring "ended"
MAX_WAIT_SECS=$((7 * 3600))   # absolute cap (7h) -- stays under the 8h watchdog

GIT_AUTHOR_NAME="results-watcher"
GIT_AUTHOR_EMAIL="results-watcher@microbe-foundation.local"

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
log() { printf '%s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }

log "==== results_watcher starting ===="
log "repo=$REPO results_branch=$RESULTS_BRANCH cwd=$(pwd)"
log "poll=${POLL_SECS}s empty_limit=$EMPTY_LIMIT max_wait=${MAX_WAIT_SECS}s"

# ----------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------
if [ ! -f "$TOKEN_FILE" ]; then
    log "FATAL: token file $TOKEN_FILE not found. Cannot push. Exiting (box left alive)."
    exit 1
fi
chmod 600 "$TOKEN_FILE" 2>/dev/null || true

if ! command -v git >/dev/null 2>&1; then
    log "FATAL: git not found on PATH. Exiting (box left alive)."
    exit 1
fi

if [ ! -d .git ]; then
    log "FATAL: cwd $(pwd) is not a git repo (no .git). Exiting (box left alive)."
    exit 1
fi

# ----------------------------------------------------------------------------
# Completion detection
#   SUCCESS : both JSONs exist.
#   ENDED   : no finetune_lora.py AND no torchrun procs for EMPTY_LIMIT checks in
#             a row, AND at least the frozen JSON exists (so the brief inter-arm
#             gap, where no training procs run for a few seconds, cannot trigger).
#   TIMEOUT : absolute max wait elapsed -> push whatever exists.
# ----------------------------------------------------------------------------
training_running() {
    # Match either the script name or a torchrun launcher. pgrep -f scans full
    # cmdline. Exclude this watcher itself defensively (it never matches anyway).
    if pgrep -f 'finetune_lora\.py' >/dev/null 2>&1; then return 0; fi
    if pgrep -f 'torchrun' >/dev/null 2>&1; then return 0; fi
    return 1
}

start_ts=$(date +%s)
empty_streak=0
outcome=""

while :; do
    now=$(date +%s)
    elapsed=$(( now - start_ts ))

    have_frozen=0; [ -f "$FROZEN_JSON" ] && have_frozen=1
    have_lora=0;   [ -f "$LORA_JSON" ]   && have_lora=1

    # --- SUCCESS: both result JSONs present ---
    if [ "$have_frozen" -eq 1 ] && [ "$have_lora" -eq 1 ]; then
        log "DETECT: both result JSONs present -> SUCCESS (elapsed ${elapsed}s)"
        outcome="success"
        break
    fi

    # --- training process accounting ---
    if training_running; then
        if [ "$empty_streak" -ne 0 ]; then
            log "training procs present again; resetting empty_streak (was $empty_streak)"
        fi
        empty_streak=0
    else
        empty_streak=$(( empty_streak + 1 ))
    fi

    log "poll: frozen=$have_frozen lora=$have_lora training=$([ "$empty_streak" -eq 0 ] && echo yes || echo no) empty_streak=$empty_streak/$EMPTY_LIMIT elapsed=${elapsed}s"

    # --- ENDED (crash / partial): procs gone long enough AND frozen arm done ---
    # Requiring frozen JSON guards against the seconds-long inter-arm gap where
    # no training procs exist between the frozen and lora arms.
    if [ "$empty_streak" -ge "$EMPTY_LIMIT" ] && [ "$have_frozen" -eq 1 ]; then
        log "DETECT: no training procs for $empty_streak consecutive checks and frozen JSON present -> run ENDED (crash/partial)"
        outcome="ended"
        break
    fi

    # If procs are gone but frozen JSON not yet written, the run likely crashed
    # very early (before any result). Still bail out once the empty streak is well
    # past the limit so we never hang forever pre-frozen, but give it more slack.
    if [ "$empty_streak" -ge $(( EMPTY_LIMIT * 3 )) ] && [ "$have_frozen" -eq 0 ]; then
        log "DETECT: no training procs for $empty_streak checks and NO frozen JSON -> early crash, nothing to push"
        outcome="ended"
        break
    fi

    # --- TIMEOUT: absolute cap ---
    if [ "$elapsed" -ge "$MAX_WAIT_SECS" ]; then
        log "DETECT: max wait ${MAX_WAIT_SECS}s reached -> proceeding to push-what-exists"
        outcome="timeout"
        break
    fi

    sleep "$POLL_SECS"
done

log "loop exited: outcome=$outcome frozen=$([ -f "$FROZEN_JSON" ] && echo 1 || echo 0) lora=$([ -f "$LORA_JSON" ] && echo 1 || echo 0)"

# ----------------------------------------------------------------------------
# Push results
# ----------------------------------------------------------------------------
push_results() {
    # Returns 0 on a successful push (or "nothing to commit but branch pushed"),
    # non-zero on any failure.
    local rc

    git config user.email "$GIT_AUTHOR_EMAIL"
    git config user.name  "$GIT_AUTHOR_NAME"

    # runs/ is gitignored -> force-add.
    git add -f "$RESULTS_DIR" 2>>"$LOG"

    if git diff --cached --quiet; then
        log "nothing staged to commit (no JSONs found in $RESULTS_DIR)."
        # Still nothing to push -> treat as failure-to-deliver so we DON'T
        # terminate; a human should investigate.
        return 2
    fi

    git commit -m "results: LoRA family run ($RESULTS_BRANCH)" >>"$LOG" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        log "ERROR: git commit failed (rc=$rc)."
        return "$rc"
    fi
    log "committed results."

    # --- one-shot GIT_ASKPASS so the token never touches .git/config or argv ---
    local askpass
    askpass="$(mktemp)" || { log "ERROR: mktemp failed"; return 1; }
    chmod 700 "$askpass"
    # Git calls the askpass helper twice: once for Username, once for Password.
    # We answer "x-access-token" for the username and the PAT for the password.
    cat > "$askpass" <<'ASKPASS'
#!/usr/bin/env bash
case "$1" in
    *Username*) printf '%s' "x-access-token" ;;
    *Password*) cat "$HOME/.gh_token" ;;
    *)          cat "$HOME/.gh_token" ;;
esac
ASKPASS

    log "pushing HEAD -> $RESULTS_BRANCH on $REPO (GIT_ASKPASS, token not in argv/config)"
    GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
        git push "https://github.com/${REPO}.git" "HEAD:refs/heads/${RESULTS_BRANCH}" >>"$LOG" 2>&1
    rc=$?

    rm -f "$askpass"

    if [ "$rc" -eq 0 ]; then
        log "push SUCCEEDED -> branch $RESULTS_BRANCH"
    else
        log "ERROR: push FAILED (rc=$rc)."
    fi
    return "$rc"
}

push_results
PUSH_RC=$?
log "push_results returned rc=$PUSH_RC"

# ----------------------------------------------------------------------------
# Terminate ONLY on successful push
# ----------------------------------------------------------------------------
if [ "$PUSH_RC" -eq 0 ]; then
    log "results safely on GitHub (branch $RESULTS_BRANCH). Proceeding to self-terminate."
    if [ -f "$RUN_ENV" ]; then
        # shellcheck disable=SC1090
        source "$RUN_ENV"
    else
        log "WARNING: $RUN_ENV not found; cannot read INSTANCE_ID/LAMBDA_API_KEY. Box left for watchdog."
        exit 0
    fi

    if [ -z "${INSTANCE_ID:-}" ] || [ -z "${LAMBDA_API_KEY:-}" ]; then
        log "WARNING: INSTANCE_ID or LAMBDA_API_KEY empty after sourcing $RUN_ENV. Box left for watchdog."
        exit 0
    fi

    log "terminating instance $INSTANCE_ID via scripts/lambda_launch.sh ..."
    if bash scripts/lambda_launch.sh terminate "$INSTANCE_ID" >>"$LOG" 2>&1; then
        log "terminate command sent for $INSTANCE_ID. Goodbye."
    else
        log "WARNING: terminate command returned non-zero; the 8h watchdog remains as backstop."
    fi
    exit 0
else
    log "############################################################"
    log "# PUSH DID NOT SUCCEED (rc=$PUSH_RC). NOT terminating box.  #"
    log "# Results must be rescued MANUALLY, e.g. from your laptop:  #"
    log "#   scp -r ubuntu@<box-ip>:~/microbe-foundation/runs/lora ./#"
    log "# Box is left alive; the 8h watchdog will terminate it.     #"
    log "############################################################"
    exit 1
fi
