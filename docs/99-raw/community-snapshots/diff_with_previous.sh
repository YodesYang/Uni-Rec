#!/bin/bash
# Diff the latest two community snapshots so we only review what changed.
#
# Usage:
#   cd docs/99-raw/community-snapshots/
#   ./diff_with_previous.sh
#
# Output:
#   - New questions (highlighted if organization == "Administrators")
#   - Existing questions with new replies (views/ans count changes)
#   - Removed questions (rare)
#
# Requirements: jq installed (brew install jq).

set -euo pipefail

cd "$(dirname "$0")"

# Find the two latest .json snapshots (sorted by filename = sorted by timestamp).
mapfile -t SNAPSHOTS < <(ls -1 lexiang_questions_*.json 2>/dev/null | sort)
N=${#SNAPSHOTS[@]}

if [ "$N" -lt 2 ]; then
    echo "[diff] Need at least 2 .json snapshots in $(pwd) to diff."
    echo "[diff] Found: $N snapshot(s)."
    exit 0
fi

PREV="${SNAPSHOTS[$((N-2))]}"
CURR="${SNAPSHOTS[$((N-1))]}"

echo "[diff] Previous: $PREV"
echo "[diff] Current : $CURR"
echo "----------------------------------------------------------------"

# Extract (id -> {title, organization, views, ans}) maps and diff them.
# We assume each entry has fields: id (or url), title, organization, views, ans.
# Adjust jq paths if the JSON schema differs from this assumption.

PREV_MAP=$(jq -c '
  if type == "array" then
    map({(.id // .url): {title, organization, views, ans}}) | add
  else
    .questions | map({(.id // .url): {title, organization, views, ans}}) | add
  end // {}
' "$PREV")

CURR_MAP=$(jq -c '
  if type == "array" then
    map({(.id // .url): {title, organization, views, ans}}) | add
  else
    .questions | map({(.id // .url): {title, organization, views, ans}}) | add
  end // {}
' "$CURR")

# New questions: in current but not in previous.
echo "[diff] === NEW questions ==="
jq -nr --argjson p "$PREV_MAP" --argjson c "$CURR_MAP" '
  ($c | keys) - ($p | keys) | .[] as $k |
  $c[$k] | "\(if .organization == "Administrators" then "[OFFICIAL] " else "" end)\(.title)  (views=\(.views), ans=\(.ans))"
' || echo "  (none)"

# Existing questions with new replies (ans count went up).
echo ""
echo "[diff] === Updated questions (more replies) ==="
jq -nr --argjson p "$PREV_MAP" --argjson c "$CURR_MAP" '
  ($p | keys) as $pk | ($c | keys) as $ck |
  ($pk - ($pk - $ck)) | .[] as $k |
  if ($c[$k].ans // 0) > ($p[$k].ans // 0) then
    "\($c[$k].title)  (ans: \($p[$k].ans) -> \($c[$k].ans))"
  else empty end
' || echo "  (none)"

# Removed questions.
echo ""
echo "[diff] === REMOVED questions ==="
jq -nr --argjson p "$PREV_MAP" --argjson c "$CURR_MAP" '
  ($p | keys) - ($c | keys) | .[] as $k |
  $p[$k] | "\(.title)  (was views=\(.views), ans=\(.ans))"
' || echo "  (none)"

echo ""
echo "----------------------------------------------------------------"
echo "[diff] Done. Merge anything valuable into docs/10-research/community-intel.md."
