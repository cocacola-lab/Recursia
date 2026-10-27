#!/usr/bin/env bash
# Fetch the pinned third-party sources (third_party.lock) into cap/third_party/ and apply the Contact-GraspNet compatibility patch.
# Idempotent: an existing checkout at the pinned commit is kept; a checkout at another commit is an error (never overwritten).
# Usage (from recursia/sim): bash scripts/fetch_third_party.sh
set -euo pipefail
SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SIM_ROOT"
while IFS=$'\t' read -r path url commit; do
  [[ -z "${path:-}" || "$path" == \#* ]] && continue
  if [[ -e "$path/.git" ]]; then
    have=$(git -C "$path" rev-parse HEAD)
    [[ "$have" == "$commit" ]] || { echo "ERROR: $path is at $have, expected $commit (move it away to refetch)"; exit 1; }
    echo "ok   $path @ ${commit:0:12}"
    continue
  fi
  echo "get  $path @ ${commit:0:12}  ($url)"
  mkdir -p "$(dirname "$path")"
  git clone --quiet --filter=blob:none --no-checkout "$url" "$path"
  git -C "$path" fetch --quiet origin "$commit" 2>/dev/null || true   # commits on non-default branches
  git -C "$path" -c advice.detachedHead=false checkout --quiet "$commit"
  [[ "$(git -C "$path" rev-parse HEAD)" == "$commit" ]] || { echo "ERROR: $path checkout failed"; exit 1; }
done < third_party.lock
bash scripts/apply_contact_graspnet_patch.sh
echo "THIRD_PARTY_OK"
