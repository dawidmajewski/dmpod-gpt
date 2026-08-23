#!/usr/bin/env bash

_dmpod_banner_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_dmpod_workspace="${DMPOD_WORKSPACE:-/workspace}"

cat "$_dmpod_banner_root/banner.txt"
if [[ ! -f "$_dmpod_workspace/.dmpod/config.toml" ]]; then
  printf '\nRun "dmpod-setup" to set up Weights & Biases and Hugging Face login.\n'
fi

unset _dmpod_banner_root _dmpod_workspace
