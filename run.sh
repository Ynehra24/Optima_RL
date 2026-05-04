#!/usr/bin/env bash
# =============================================================================
# run.sh  —  Optima RL Project: Full Automated Pipeline
# =============================================================================
# Targets: Ubuntu 22.04 (docker container, clean environment)
#
# What this script does:
#   1. Install required system packages
#   2. Create and activate a Python virtual environment
#   3. Install all Python dependencies
#   4. Run Phase 1: Airline Hold-or-Not-Hold RL (A2C, DQN, AC, DDPG)
#   5. Run Phase 2: Logistics Cargo Hold-or-Not-Hold RL (multi-hub)
#   6. Run Phase 3: DAG Scheduling Hold-or-Not-Hold RL + benchmark check
#   7. Collect all outputs into a single top-level results/ directory
# =============================================================================

set -euo pipefail   # Exit on any error, unset variable, or pipe failure

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${CYAN}[run.sh]${RESET} $*"; }
ok()   { echo -e "${GREEN}[✔]${RESET} $*"; }
warn() { echo -e "${YELLOW}[⚠]${RESET} $*"; }
fail() { echo -e "${RED}[✘]${RESET} $*"; exit 1; }

# ── Resolve project root (always relative) ───────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"   # All paths below are relative to the repo root

# ── Parse command-line arguments ──────────────────────────────────────────────
EPISODES=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

VENV_DIR="./venv"
RESULTS_ROOT="./results"
LOGS_DIR="$RESULTS_ROOT/logs"
mkdir -p "$LOGS_DIR"

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "============================================================"
echo "   Optima RL Project — Full Automated Pipeline"
echo "   $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo -e "${RESET}"

# =============================================================================
# STEP 1 — System dependencies (Ubuntu 22.04)
# =============================================================================
log "STEP 1: Installing system packages..."

apt-get update -qq 2>&1 | tail -3
apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libpcap-dev \
    2>&1 | tail -5

ok "System packages installed."

# =============================================================================
# STEP 2 — Python virtual environment
# =============================================================================
log "STEP 2: Creating virtual environment at $VENV_DIR ..."

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

ok "Virtual environment activated: $(python --version)"

# =============================================================================
# STEP 3 — Python dependencies
# =============================================================================
log "STEP 3: Installing Python dependencies..."

pip install --upgrade pip --quiet
pip install --quiet \
    "numpy>=1.26" \
    "matplotlib>=3.8" \
    "gymnasium>=0.29" \
    "pandas>=2.1" \
    "scapy>=2.5"

ok "Python dependencies installed."
pip list | grep -E "numpy|matplotlib|gymnasium|pandas|scapy" | \
    awk '{printf "  %-20s %s\n", $1, $2}'

# =============================================================================
# STEP 4 — Phase 1: Airline Network Simulator (Hold-or-Not-Hold)
# =============================================================================
echo ""
echo -e "${BOLD}============================================================${RESET}"
log "STEP 4: Running Phase 1 — Airline RL Training"
echo -e "${BOLD}============================================================${RESET}"

PHASE1_TRAIN="./phase-1/algoImplementation/train.py"
PHASE1_RESULTS="./phase-1/algoImplementation/results"
mkdir -p "$PHASE1_RESULTS"

if [ -n "$EPISODES" ]; then
    log "Training all agents (A2C, DQN, AC, DDPG) — $EPISODES episodes each..."
    python "$PHASE1_TRAIN" \
        --algo all \
        --episodes "$EPISODES" \
        2>&1 | tee "$LOGS_DIR/phase1_training.log"
else
    log "Training all agents (A2C, DQN, AC, DDPG) — default 25 episodes each..."
    python "$PHASE1_TRAIN" \
        --algo all \
        2>&1 | tee "$LOGS_DIR/phase1_training.log"
fi

ok "Phase 1 training complete."
log "Phase 1 output files:"
ls -lh "$PHASE1_RESULTS/" 2>/dev/null | awk '{print "  " $0}'

# Copy Phase 1 artefacts to top-level results
mkdir -p "$RESULTS_ROOT/phase1"
cp -r "$PHASE1_RESULTS"/. "$RESULTS_ROOT/phase1/"
ok "Phase 1 results saved → $RESULTS_ROOT/phase1/"

# =============================================================================
# STEP 5 — Phase 2: Logistics Cargo Simulator (Multi-Hub)
# =============================================================================
echo ""
echo -e "${BOLD}============================================================${RESET}"
log "STEP 5: Running Phase 2 — Logistics RL Training (Multi-Hub)"
echo -e "${BOLD}============================================================${RESET}"

PHASE2_TRAIN="./phase-2/algoImplementation/train.py"
PHASE2_RESULTS="./phase-2/algoImplementation/results"
mkdir -p "$PHASE2_RESULTS"

if [ -n "$EPISODES" ]; then
    log "Training all agents (A2C, DQN, AC, DDPG) — $EPISODES episodes, multi-hub mode..."
    python "$PHASE2_TRAIN" \
        --algo all \
        --multi-hub \
        --episodes "$EPISODES" \
        2>&1 | tee "$LOGS_DIR/phase2_training.log"
else
    log "Training all agents (A2C, DQN, AC, DDPG) — default 25 episodes, multi-hub mode..."
    python "$PHASE2_TRAIN" \
        --algo all \
        --multi-hub \
        2>&1 | tee "$LOGS_DIR/phase2_training.log"
fi

ok "Phase 2 training complete."
log "Phase 2 output files:"
ls -lh "$PHASE2_RESULTS/" 2>/dev/null | awk '{print "  " $0}'

# Copy Phase 2 artefacts to top-level results
mkdir -p "$RESULTS_ROOT/phase2"
cp -r "$PHASE2_RESULTS"/. "$RESULTS_ROOT/phase2/"
ok "Phase 2 results saved → $RESULTS_ROOT/phase2/"

# =============================================================================
# STEP 6 — Phase 3: DAG Scheduling Simulator (Hold-or-Not-Hold)
# =============================================================================
echo ""
echo -e "${BOLD}============================================================${RESET}"
log "STEP 6: Running Phase 3 — DAG Scheduling RL Training (standard preset)"
echo -e "${BOLD}============================================================${RESET}"

PHASE3_TRAIN="./phase-3/algoImplementation/train.py"
PHASE3_BENCH="./phase-3/algoImplementation/benchmark_check.py"
PHASE3_RESULTS="./phase-3/algoImplementation/results"
PHASE3_LOGS="./phase-3/algoImplementation/training_logs"
mkdir -p "$PHASE3_RESULTS" "$PHASE3_LOGS"

RUN_NAME="eval_run"

if [ -n "$EPISODES" ]; then
    log "Training all agents (A2C, DQN, AC, DDPG) — $EPISODES episodes (standard preset)..."
    python "$PHASE3_TRAIN" \
        --algo all \
        --preset standard \
        --run-name "$RUN_NAME" \
        --episodes "$EPISODES" \
        2>&1 | tee "$LOGS_DIR/phase3_training.log"
else
    log "Training all agents (A2C, DQN, AC, DDPG) — standard preset (30 episodes)..."
    python "$PHASE3_TRAIN" \
        --algo all \
        --preset standard \
        --run-name "$RUN_NAME" \
        2>&1 | tee "$LOGS_DIR/phase3_training.log"
fi

ok "Phase 3 training complete."

# =============================================================================
# STEP 7 — Phase 3: Benchmark / Evaluation Check
# =============================================================================
echo ""
log "STEP 7: Running Phase 3 benchmark check..."

PHASE3_SUMMARY="$PHASE3_RESULTS/${RUN_NAME}_summary.json"
BENCHMARK_LOG="$PHASE3_LOGS/${RUN_NAME}_benchmark.log"

# benchmark_check exits with code 1 if ALL metrics fail; we capture but don't abort
set +e
python "$PHASE3_BENCH" \
    --summary  "$PHASE3_SUMMARY" \
    --all-agents \
    --output   "$BENCHMARK_LOG" \
    2>&1 | tee "$LOGS_DIR/phase3_benchmark.log"
BENCH_EXIT=$?
set -e

if [ $BENCH_EXIT -eq 0 ]; then
    ok "Phase 3 benchmark: PASS — RL agents meet or exceed baselines."
else
    warn "Phase 3 benchmark: some agents did not outperform all baselines."
    warn "This is expected for shorter training runs (standard preset)."
    warn "See $BENCHMARK_LOG for details."
fi

log "Phase 3 output files:"
ls -lh "$PHASE3_RESULTS/" 2>/dev/null | awk '{print "  " $0}'

# Copy Phase 3 artefacts to top-level results
mkdir -p "$RESULTS_ROOT/phase3"
cp -r "$PHASE3_RESULTS"/. "$RESULTS_ROOT/phase3/"
[ -f "$BENCHMARK_LOG" ] && cp "$BENCHMARK_LOG" "$RESULTS_ROOT/phase3/"
ok "Phase 3 results saved → $RESULTS_ROOT/phase3/"

# =============================================================================
# STEP 8 — Consolidated summary
# =============================================================================
echo ""
echo -e "${BOLD}============================================================${RESET}"
log "STEP 8: Writing consolidated summary..."

SUMMARY_FILE="$RESULTS_ROOT/run_summary.txt"
{
    echo "Optima RL Project — Run Summary"
    echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "==============================="
    echo " Phase 1 — Airline RL"
    echo "==============================="
    if [ -f "$RESULTS_ROOT/phase1/summary.json" ]; then
        python -c "
import json, sys
d = json.load(open('$RESULTS_ROOT/phase1/summary.json'))
methods = ['no_hold','heuristic_15','heuristic_30','a2c','dqn','ac','ddpg']
print(f\"  {'Method':<14} {'Missed PAX':>11} {'OTP%':>7} {'Holds%':>7}\")
print('  ' + '-'*43)
for m in methods:
    if m not in d or m == '_deltas': continue
    r = d[m]
    print(f\"  {m:<14} {r.get('missed_connections',0):>11.0f} {r.get('OTP',0):>7.1f}% {r.get('holds_pct',0):>6.1f}%\")
" 2>/dev/null || echo "  (see phase1/summary.json)"
    fi
    echo ""
    echo "==============================="
    echo " Phase 2 — Logistics RL"
    echo "==============================="
    if [ -f "$RESULTS_ROOT/phase2/summary.json" ]; then
        python -c "
import json
d = json.load(open('$RESULTS_ROOT/phase2/summary.json'))
methods = ['no_hold','heuristic_15','heuristic_30','a2c','dqn','ac','ddpg']
print(f\"  {'Method':<14} {'Miss Rate':>10} {'SLA%':>7} {'Holds%':>7}\")
print('  ' + '-'*43)
for m in methods:
    if m not in d: continue
    r = d[m]
    print(f\"  {m:<14} {r.get('missed_rate',0):>9.2%} {r.get('SLA_compliance',0):>7.1f}% {r.get('holds_pct',0):>6.1f}%\")
" 2>/dev/null || echo "  (see phase2/summary.json)"
    fi
    echo ""
    echo "==============================="
    echo " Phase 3 — DAG Scheduling RL"
    echo "==============================="
    if [ -f "$RESULTS_ROOT/phase3/summary.json" ]; then
        python -c "
import json
d = json.load(open('$RESULTS_ROOT/phase3/summary.json'))
ev = d.get('evaluation', d)
methods = ['no_hold','heuristic','gpu_guard','a2c','dqn','ac','ddpg']
print(f\"  {'Method':<12} {'Done%':>8} {'Evict%':>8} {'Stalls':>8} {'AvgReward':>10}\")
print('  ' + '-'*52)
for m in methods:
    if m not in ev: continue
    r = ev[m]
    print(f\"  {m:<12} {r.get('completed_pct',0):>7.1f} {r.get('evicted_pct',0):>7.1f} {r.get('pipeline_stalls',0):>8.1f} {r.get('avg_reward',0):>10.4f}\")
" 2>/dev/null || echo "  (see phase3/summary.json)"
    fi
    echo ""
    echo "==============================="
    echo " Artefacts"
    echo "==============================="
    echo "  Logs      : $LOGS_DIR/"
    echo "  Phase 1   : $RESULTS_ROOT/phase1/"
    echo "  Phase 2   : $RESULTS_ROOT/phase2/"
    echo "  Phase 3   : $RESULTS_ROOT/phase3/"
} | tee "$SUMMARY_FILE"

ok "Summary written to $SUMMARY_FILE"

# =============================================================================
# DONE
# =============================================================================
echo ""
echo -e "${GREEN}${BOLD}============================================================"
echo "  All phases completed successfully."
echo "  Results are in: $RESULTS_ROOT/"
echo -e "============================================================${RESET}"
