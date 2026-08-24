#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="${IMAGE_NAME:-dawidmkrk/dmpod-gpt:local}"
nanogpt_revision="${NANOGPT_REVISION:-$(tr -d '[:space:]' < "$project_dir/NANOGPT_REVISION")}"
dmpod_version="$(tr -d '[:space:]' < "$project_dir/DMPOD_VERSION")"

docker build --platform linux/amd64 \
  --build-arg "NANOGPT_REVISION=$nanogpt_revision" \
  --build-arg "DMPOD_VERSION=$dmpod_version" \
  --tag "$image_name" "$project_dir"

echo "Built DMPod $dmpod_version as $image_name with nanoGPT $nanogpt_revision"
