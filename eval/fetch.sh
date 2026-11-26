#!/usr/bin/env bash
# Materializes the eval fixtures' trees for `diffagent-eval context`: the only
# networked eval step, read-only on GitHub.
#
#   eval/fetch.sh [FIXTURE…]      (all labeled fixtures by default)
#
# Works only on labels that name real repositories and commits, so not on the
# synthetic corpus. Needs bash, git, gh (authenticated), jq and go. The cache sits
# outside the repo, since the trees may be private code:
# ${DIFFAGENT_EVAL_CACHE:-$HOME/.cache/diffagent/eval}
#   repos/<owner>/<repo>.git        one bare repo per repo, shallow fetches by SHA
#   trees/<fixture>/                the tree at the planted base_sha (patch applied)
#                                   or the history head_sha; symlinks as plain files
#   trees/<fixture>.files.json      the PR's files: planted, the fixture's; history,
#                                   every file of the compare
#   gomod/                          GOMODCACHE for the Go trees
# A fixture whose tree exists is skipped; delete the tree to fetch it again.
set -euo pipefail

corpus=$(cd "$(dirname "$0")" && pwd)/synthetic
cache=${DIFFAGENT_EVAL_CACHE:-$HOME/.cache/diffagent/eval}
labels=$(jq -s 'add' "$corpus/labels.history.json" "$corpus/labels.planted.json")
mkdir -p "$cache/repos" "$cache/trees" "$cache/gomod"

# gh hands git the token; no other helper is asked and nothing is stored.
helper='!gh auth git-credential'

if [ $# -eq 0 ]; then
  # Fixture names have no spaces or globs.
  # shellcheck disable=SC2046
  set -- $(jq -r 'keys[]' <<<"$labels")
fi

for name in "$@"; do
  fx=$(jq -c --arg n "$name" '.[$n] // empty' <<<"$labels")
  [ -n "$fx" ] || { echo "$name: not a labeled fixture" >&2; exit 1; }
  tree="$cache/trees/$name"
  if [ -d "$tree" ]; then
    echo "$name: cached"
    continue
  fi
  repo=$(jq -r .repo <<<"$fx")
  kind=$(jq -r .kind <<<"$fx")
  base=$(jq -r .base_sha <<<"$fx")
  sha=$base
  [ "$kind" = planted ] || sha=$(jq -r .head_sha <<<"$fx")

  bare="$cache/repos/$repo.git"
  if [ ! -d "$bare" ]; then
    git init -q --bare "$bare"
    git -C "$bare" remote add origin "https://github.com/$repo.git"
  fi
  if ! git -C "$bare" cat-file -e "$sha^{commit}" 2>/dev/null; then
    git -C "$bare" -c credential.helper= -c credential.helper="$helper" \
      fetch -q --depth 1 --no-tags origin "$sha"
  fi

  tmp="$tree.tmp"
  rm -rf "$tmp" "$tmp.index"
  mkdir -p "$tmp"
  GIT_INDEX_FILE="$tmp.index" git --git-dir="$bare" --work-tree="$tmp" read-tree "$sha"
  GIT_INDEX_FILE="$tmp.index" git --git-dir="$bare" --work-tree="$tmp" \
    -c core.symlinks=false checkout-index -a -f
  rm -f "$tmp.index"

  fixture="$corpus/fixtures/$name.json"
  if [ "$kind" = planted ]; then
    jq -r '.files[] | "diff --git a/\(.Path) b/\(.Path)\n"
      + "--- \(if .Status == "added" then "/dev/null" else "a/\(.Path)" end)\n"
      + "+++ \(if .Status == "removed" then "/dev/null" else "b/\(.Path)" end)\n"
      + .Patch' "$fixture" >"$tmp.patch"
    # The ceiling keeps git apply from finding a repo above the tree.
    (cd "$tmp" && export GIT_CEILING_DIRECTORIES="$cache/trees" &&
      git apply --check "$tmp.patch" && git apply "$tmp.patch") ||
      { echo "$name: patch does not apply" >&2; exit 1; }
    rm -f "$tmp.patch"
    jq '[.files[] | {Path, Status, Patch}]' "$fixture" >"$tree.files.json"
  else
    gh api "repos/$repo/compare/$base...$sha" |
      jq '[.files[] | {Path: .filename, Status: .status, Patch: (.patch // "")}]' \
        >"$tree.files.json"
  fi

  if [ -f "$tmp/go.mod" ]; then
    # The repo owner's modules are private and come over git with gh's helper; the
    # token stays out of argv.
    (cd "$tmp" && GOMODCACHE="$cache/gomod" GOTOOLCHAIN=local \
      GOPRIVATE="github.com/${repo%%/*}" GIT_TERMINAL_PROMPT=0 \
      GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=credential.helper GIT_CONFIG_VALUE_0='' \
      GIT_CONFIG_KEY_1=credential.helper GIT_CONFIG_VALUE_1="$helper" \
      go mod download) || { echo "$name: go mod download failed" >&2; exit 1; }
  fi
  mv "$tmp" "$tree"
  echo "$name: $(jq length "$tree.files.json") files at ${sha:0:7}"
done
