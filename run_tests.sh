#!/usr/bin/env bash
# =============================================================================
#  run_tests.sh — one-command test suite for the Quantum Network Defense project
#
#  Usage:
#     ./run_tests.sh            # quick suite   (~2–3 min)
#     ./run_tests.sh --full     # + original legacy code, real Cirq circuit
#                               #   sampling, a 16-qubit run   (~8–10 min)
#     ./run_tests.sh --help
#
#  Every step prints PASS / FAIL with its runtime. All run outputs go to
#  outputs/test_runs/ (the folder is recreated each time). The script exits
#  with status 1 if any step failed, so it can be used in CI.
# =============================================================================

set -u
cd "$(dirname "$0")"

PY=./venv/bin/python
OUT=outputs/test_runs
FULL=0

case "${1:-}" in
  --full) FULL=1 ;;
  --help|-h) sed -n '2,13p' "$0"; exit 0 ;;
  "") ;;
  *) echo "unknown option: $1 (use --full or --help)"; exit 2 ;;
esac

GREEN=$'\e[32m'; RED=$'\e[31m'; BOLD=$'\e[1m'; DIM=$'\e[2m'; RESET=$'\e[0m'
PASSED=0; FAILED=0; FAILED_NAMES=()

rm -rf "$OUT"; mkdir -p "$OUT"
LOG="$OUT/test_log.txt"; : > "$LOG"

# step "<name>" <command...>   — run a command, record PASS/FAIL and time
step() {
  local name="$1"; shift
  printf "  %-62s" "$name"
  local t0=$SECONDS
  if { echo "===== $name"; "$@"; } >>"$LOG" 2>&1; then
    printf "${GREEN}PASS${RESET} ${DIM}(%ds)${RESET}\n" $((SECONDS - t0))
    PASSED=$((PASSED + 1))
  else
    printf "${RED}FAIL${RESET} ${DIM}(%ds)${RESET}\n" $((SECONDS - t0))
    FAILED=$((FAILED + 1)); FAILED_NAMES+=("$name")
  fi
}

# check <results.json> <python boolean expression over r (run 0) and s (summary)>
check() {
  "$PY" - "$1" "$2" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
r, s = d["runs"][0], d["summary"]
ok = eval(sys.argv[2])
print("check:", sys.argv[2], "->", ok)
sys.exit(0 if ok else 1)
EOF
}

run_main() {  # run_main <out-subdir> <main.py args...>
  local dir="$OUT/$1"; shift
  "$PY" main.py --quiet --out-dir "$dir" "$@"
}

echo "${BOLD}Quantum Network Defense — test suite$([ $FULL = 1 ] && echo ' (full)')${RESET}"
echo "${DIM}log: $LOG${RESET}"

# ── 0. Environment ───────────────────────────────────────────────────────────
echo; echo "${BOLD}[0] Environment${RESET}"
step "virtualenv python exists"             test -x "$PY"
step "output.csv present"                    test -f output.csv
step "required packages import"              "$PY" -c "import cirq, sklearn, numpy, pandas, scipy, matplotlib"

# ── 1. Unit tests ────────────────────────────────────────────────────────────
echo; echo "${BOLD}[1] Unit tests (tests/test_core.py)${RESET}"
step "unit tests (19)"                    "$PY" -m unittest discover -s tests

# ── 2. Original protocol vs improved protocol ────────────────────────────────
echo; echo "${BOLD}[2] Original vs improved protocol (binary, 8 qubits)${RESET}"
step "--legacy run completes"                run_main legacy --legacy --n-qubits 8
step "  legacy: >20% test rows leak from train" \
     check "$OUT/legacy/results.json" 'r["data"]["test_rows_with_exact_copy_in_train"] > 0.20'
step "improved run completes"                run_main improved --n-qubits 8
step "  improved: duplicates removed (>40,000)" \
     check "$OUT/improved/results.json" 'r["data"]["duplicates_removed"] > 40000'
step "  improved: <2% test rows leak from train" \
     check "$OUT/improved/results.json" 'r["data"]["test_rows_with_exact_copy_in_train"] < 0.02'
step "  improved: QSVM accuracy > 95%" \
     check "$OUT/improved/results.json" 's["qsvm"]["accuracy"]["mean"] > 0.95'
step "  improved: alpha and C were tuned by CV" \
     check "$OUT/improved/results.json" 'len(r["qsvm_tuning"]) == 20'
step "  improved: kernel diagnostics recorded" \
     check "$OUT/improved/results.json" '"geometric_difference" in r["kernel_diagnostics"]["quantum"]'
step "  improved: per-run figure written"     test -f "$OUT/improved/results_seed42.png"

# ── 3. Multiclass ────────────────────────────────────────────────────────────
echo; echo "${BOLD}[3] Multiclass (6 classes, 8 qubits)${RESET}"
step "multiclass run completes"              run_main multiclass --mode multiclass --n-qubits 8
step "  6x6 confusion matrix, accuracy > 80%" \
     check "$OUT/multiclass/results.json" 'len(r["qsvm"]["confusion_matrix"]) == 6 and s["qsvm"]["accuracy"]["mean"] > 0.80'
step "  one-vs-rest AUC reported" \
     check "$OUT/multiclass/results.json" 'r["qsvm"]["auc"] is not None'

# ── 4. The paper's three methods ─────────────────────────────────────────────
echo; echo "${BOLD}[4] Paper methods through one pipeline${RESET}"
step "Method 1: separable Rx kernel"         run_main rx --kernel rx --n-qubits 8
step "Method 2: Rx + 1024 shots (binomial)"  run_main rx_shots --kernel rx --n-qubits 8 --sampling --shots 1024
step "Method 3: ZZ, full entanglement, 3 reps" run_main zz_full --n-qubits 6 --entanglement full --n-reps 3
step "  all three reach > 90% accuracy" bash -c "
  $PY -c \"
import json
for d in ['rx','rx_shots','zz_full']:
    a = json.load(open('$OUT/'+d+'/results.json'))['summary']['qsvm']['accuracy']['mean']
    print(d, a); assert a > 0.90
\""

# ── 5. Fast engine == Cirq ───────────────────────────────────────────────────
echo; echo "${BOLD}[5] Fast engine vs Cirq simulator (end-to-end)${RESET}"
common=(--n-qubits 8 --samples 500 --no-tune --alpha 0.4 --C 10)
step "fast backend run"                      run_main check_fast "${common[@]}" --backend fast
step "cirq backend run"                      run_main check_cirq "${common[@]}" --backend cirq
step "  identical predictions and AUC" "$PY" -c "
import json
a = json.load(open('$OUT/check_fast/results.json'))['runs'][0]['qsvm']
b = json.load(open('$OUT/check_cirq/results.json'))['runs'][0]['qsvm']
print(a['confusion_matrix'], b['confusion_matrix'], a['auc'], b['auc'])
assert a['confusion_matrix'] == b['confusion_matrix'] and abs(a['auc'] - b['auc']) < 1e-6
"

# ── 6. Multi-seed + baselines + reproducibility ──────────────────────────────
echo; echo "${BOLD}[6] Multi-seed, extra baselines, reproducibility${RESET}"
step "3 seeds with Random Forest baseline"   run_main seeds --n-qubits 6 --seeds 0 1 2 --extra-baselines
step "  mean ± std and McNemar for every seed" \
     check "$OUT/seeds/results.json" 'len(d["runs"]) == 3 and "std" in s["qsvm"]["accuracy"] and "random_forest" in s and all("p_value" in x["mcnemar_qsvm_vs_rbf"] for x in d["runs"])'
step "repeat seed 0 (reproducibility)"       run_main seeds_repeat --n-qubits 6 --seeds 0 --extra-baselines
step "  same seed -> identical results" "$PY" -c "
import json
a = json.load(open('$OUT/seeds/results.json'))['runs'][0]
b = json.load(open('$OUT/seeds_repeat/results.json'))['runs'][0]
for k in ['qsvm', 'rbf_svm', 'random_forest']:
    assert a[k]['confusion_matrix'] == b[k]['confusion_matrix'], k
assert a['qsvm']['alpha'] == b['qsvm']['alpha'] and a['qsvm']['C'] == b['qsvm']['C']
print('identical')
"

# ── 7. Full suite only ───────────────────────────────────────────────────────
if [ $FULL = 1 ]; then
  echo; echo "${BOLD}[7] Full suite: original code, real circuits, 16 qubits${RESET}"
  # the original main.py overwrites outputs/qnd_results.png — keep a copy
  [ -f outputs/qnd_results.png ] && cp outputs/qnd_results.png "$OUT/qnd_results_backup.png"
  step "original legacy/main.py runs unchanged" \
       "$PY" legacy/main.py --samples 100 --n-qubits 4 --n-reps 2
  [ -f "$OUT/qnd_results_backup.png" ] && cp "$OUT/qnd_results_backup.png" outputs/qnd_results.png
  step "real Cirq circuit sampling (256 shots)" \
       run_main cirq_sampling --backend cirq --sampling --shots 256 --samples 100 --n-qubits 4 --no-tune --alpha 0.4
  step "16-qubit ZZ, 2 reps, tuned (default config)" run_main q16 --n-qubits 16 --samples 1000
  step "  16-qubit QSVM accuracy > 95%" \
       check "$OUT/q16/results.json" 's["qsvm"]["accuracy"]["mean"] > 0.95'
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo
echo "${BOLD}Summary:${RESET} ${GREEN}${PASSED} passed${RESET}, $([ $FAILED -gt 0 ] && echo "${RED}" )${FAILED} failed${RESET}  ${DIM}(total ${SECONDS}s)${RESET}"
if [ $FAILED -gt 0 ]; then
  printf "  ${RED}✗ %s${RESET}\n" "${FAILED_NAMES[@]}"
  echo "  details: $LOG"
  exit 1
fi
echo "  outputs: $OUT/   log: $LOG"
