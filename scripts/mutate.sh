#!/usr/bin/env bash
#
# Apply one mutation to a source file, run the unit suite, and restore.
#
#   scripts/mutate.sh <file> <old text> <new text> <label>
#
# A surviving mutation is a gap in the tests. Three things make that claim
# trustworthy, and each was learned by getting it wrong:
#
# 1. `__pycache__` is cleared first. CPython decides a cached module is current
#    from the source's size and its mtime truncated to whole seconds, so a
#    mutation that preserves the byte count and is tested within a second of
#    being written runs the PREVIOUS bytecode. The suite passes, and that is
#    indistinguishable from a mutation the tests genuinely failed to catch —
#    which biases an audit toward reporting the suite as weaker than it is.
#
# 2. The mutant must parse. A syntax error and a killed mutation both print red.
#    `_LEAD_DAYS = 020` is byte-identical to `120` and is a SyntaxError in
#    Python 3, so it "failed" the suite without ever changing a behaviour.
#
# 3. The restore is verified. An earlier version restored with `git checkout`,
#    which silently does nothing for an untracked file: four mutations stacked
#    on top of each other and every result after the first measured a file
#    nobody had read. The output table looked entirely normal.
#
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

file="$1"; old="$2"; new="$3"; label="$4"
backup=$(mktemp)
cp "$file" "$backup"

python3 - "$file" "$old" "$new" <<'PY' || { cp "$backup" "$file"; rm -f "$backup"; exit 1; }
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
found = text.count(sys.argv[2])
if found != 1:
    sys.exit(f"target appears {found} times, expected exactly 1")
path.write_text(text.replace(sys.argv[2], sys.argv[3]))
PY

find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null

if ! python3 -m py_compile "$file" 2>/dev/null; then
  cp "$backup" "$file"; rm -f "$backup"
  find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
  printf '%-46s %-16s %s\n' "$label" "INVALID" "mutant does not parse; not a behaviour change"
  exit 0
fi

result=$(uv run pytest tests/unit -q -p no:cacheprovider 2>&1 | tail -1)

cp "$backup" "$file"
if ! cmp -s "$file" "$backup"; then
  echo "FATAL: could not restore $file"; rm -f "$backup"; exit 1
fi
rm -f "$backup"
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null

[ ${#old} -eq ${#new} ] && shape="size-preserving" || shape="size-changing"
printf '%-46s %-16s %s\n' "$label" "$shape" "$result"
