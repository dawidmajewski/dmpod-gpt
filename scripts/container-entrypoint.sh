#!/usr/bin/env bash
set -euo pipefail

workspace="${DMPOD_WORKSPACE:-/workspace}"
image_nanogpt="${DMPOD_IMAGE_NANOGPT:-/opt/nanogpt}"
template_root="${DMPOD_TEMPLATE_ROOT:-/opt/dmpod/workspace-template}"
nanogpt_patch="${DMPOD_NANOGPT_PATCH-/opt/dmpod/patches/nanogpt-atomic-checkpoints.patch}"
default_project="$workspace/nanogpt"
config_path="$workspace/.dmpod/config.toml"

project_root="${DMPOD_PROJECT_ROOT:-${DMPOD_NANOGPT_ROOT:-}}"
if [[ -z "$project_root" && -f "$config_path" ]]; then
  project_root="$(python3 -c '
import sys, tomllib
from pathlib import Path
with Path(sys.argv[1]).open("rb") as source:
    config = tomllib.load(source)
print(config.get("project_root") or config.get("nanogpt_root") or sys.argv[2])
' "$config_path" "$default_project")"
fi
project_root="${project_root:-$default_project}"
project_root="$(python3 -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).expanduser().resolve())' "$project_root")"
default_project="$(python3 -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve())' "$default_project")"

mkdir -p "$workspace"
if [[ "$project_root" == "$default_project" && ! -e "$project_root" ]]; then
  temporary="$(mktemp -d "$workspace/.nanogpt.init.XXXXXX")"
  cleanup() { rm -rf "$temporary"; }
  trap cleanup EXIT
  cp -a "$image_nanogpt/." "$temporary/"
  cp -a "$template_root/." "$temporary/"
  if [[ -n "$nanogpt_patch" ]]; then
    git -C "$temporary" apply "$nanogpt_patch"
  fi
  mv "$temporary" "$project_root"
  trap - EXIT
elif [[ -e "$project_root" && ! -d "$project_root" ]]; then
  echo "$project_root exists but is not a directory" >&2
  exit 1
fi

if [[ -d "$project_root" ]]; then
  cd "$project_root"
else
  cd "$workspace"
fi
exec "$@"
