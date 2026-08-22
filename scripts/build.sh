#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="${IMAGE_NAME:-dawidmkrk/dmpod-gpt:local}"
nanogpt_revision="${NANOGPT_REVISION:-$(tr -d '[:space:]' < "$project_dir/NANOGPT_REVISION")}"

docker build --platform linux/amd64 \
  --build-arg "NANOGPT_REVISION=$nanogpt_revision" \
  --tag "$image_name" "$project_dir"

echo "Built $image_name with nanoGPT $nanogpt_revision"
