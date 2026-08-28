#!/usr/bin/env bash

_dmpod_banner_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_dmpod_workspace="${DMPOD_WORKSPACE:-/workspace}"
_dmpod_default_project="$_dmpod_workspace/nanogpt"
_dmpod_config="$_dmpod_workspace/.dmpod/config.toml"
_dmpod_project="${DMPOD_PROJECT_ROOT:-${DMPOD_NANOGPT_ROOT:-}}"
if [[ -z "$_dmpod_project" && -f "$_dmpod_config" ]]; then
  _dmpod_project="$(python3 -c '
import sys, tomllib
from pathlib import Path
with Path(sys.argv[1]).open("rb") as source:
    config = tomllib.load(source)
print(config.get("project_root") or config.get("nanogpt_root") or sys.argv[2])
' "$_dmpod_config" "$_dmpod_default_project")"
fi
_dmpod_project="${_dmpod_project:-$_dmpod_default_project}"
_dmpod_project="$(python3 -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).expanduser().resolve())' "$_dmpod_project")"
_dmpod_pwd="$(pwd -P)"
_dmpod_workspace_physical="$(cd "$_dmpod_workspace" 2>/dev/null && pwd -P)"
_dmpod_home_physical="$(cd "${HOME:-/root}" 2>/dev/null && pwd -P)"

if [[ $- == *i* && -d "$_dmpod_project" ]] && {
  [[ "$_dmpod_pwd" == "$_dmpod_workspace_physical" ]] ||
    [[ "$_dmpod_pwd" == "$_dmpod_home_physical" ]]
}; then
  cd "$_dmpod_project"
fi

cat "$_dmpod_banner_root/banner.txt"
printf 'Project: %s\n' "$_dmpod_project"
if [[ ! -f "$_dmpod_workspace/.dmpod/config.toml" ]]; then
  printf '\nRun "dmpod-setup" to set up Weights & Biases and Hugging Face login.\n'
fi

unset _dmpod_banner_root _dmpod_workspace _dmpod_default_project _dmpod_config
unset _dmpod_project _dmpod_pwd
unset _dmpod_workspace_physical _dmpod_home_physical
