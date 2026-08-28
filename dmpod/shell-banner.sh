#!/usr/bin/env bash

_dmpod_banner_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_dmpod_workspace="${DMPOD_WORKSPACE:-/workspace}"
_dmpod_project="$_dmpod_workspace/nanogpt"
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

unset _dmpod_banner_root _dmpod_workspace _dmpod_project _dmpod_pwd
unset _dmpod_workspace_physical _dmpod_home_physical
