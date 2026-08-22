#!/usr/bin/env bash
set -euo pipefail

workspace="${DMPOD_WORKSPACE:-/workspace}"
image_nanogpt="${DMPOD_IMAGE_NANOGPT:-/opt/nanogpt}"
template_root="${DMPOD_TEMPLATE_ROOT:-/opt/dmpod/workspace-template}"
nanogpt_patch="${DMPOD_NANOGPT_PATCH-/opt/dmpod/patches/nanogpt-atomic-checkpoints.patch}"
workspace_nanogpt="$workspace/nanogpt"

mkdir -p "$workspace"
if [[ ! -e "$workspace_nanogpt" ]]; then
  temporary="$(mktemp -d "$workspace/.nanogpt.init.XXXXXX")"
  cleanup() { rm -rf "$temporary"; }
  trap cleanup EXIT
  cp -a "$image_nanogpt/." "$temporary/"
  cp -a "$template_root/." "$temporary/"
  if [[ -n "$nanogpt_patch" ]]; then
    git -C "$temporary" apply "$nanogpt_patch"
  fi
  mv "$temporary" "$workspace_nanogpt"
  trap - EXIT
elif [[ ! -d "$workspace_nanogpt" ]]; then
  echo "$workspace_nanogpt exists but is not a directory" >&2
  exit 1
fi

cd "$workspace_nanogpt"
exec "$@"
