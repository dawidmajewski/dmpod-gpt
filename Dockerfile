FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851 AS runtime

ARG NANOGPT_REPOSITORY=https://github.com/karpathy/nanoGPT.git
ARG NANOGPT_REVISION=3adf61e154c3fe3fca428ad6bc3818b27a3b8291
ARG DMPOD_VERSION
ARG CODEX_VERSION=0.149.0
ARG CODEX_SHA256=1c08ba262820b78d49ea7a93f326b6b430b72e5fe46830e433edef12e5123244
ARG CLAUDE_VERSION=2.1.240
ARG CLAUDE_SHA256=1386169da77de19a655f07a86ab80f5775983a50eb0c9c27a7daf16e7320322d

LABEL org.opencontainers.image.title="dmpod-gpt" \
      org.opencontainers.image.version="${DMPOD_VERSION}" \
      org.opencontainers.image.description="Persistent nanoGPT development and training environment for RunPod" \
      org.opencontainers.image.source="https://github.com/dawidmajewski/dmpod-gpt" \
      io.dmpod.nanogpt.revision="${NANOGPT_REVISION}" \
      io.openai.codex.version="${CODEX_VERSION}" \
      io.anthropic.claude-code.version="${CLAUDE_VERSION}"

ENV DMPOD_VERSION=${DMPOD_VERSION} \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/cache/huggingface \
    HF_HUB_CACHE=/workspace/cache/huggingface/hub \
    WANDB_DIR=/workspace/logs/wandb \
    TORCH_HOME=/workspace/cache/torch \
    TOKENIZERS_PARALLELISM=false \
    NCCL_DEBUG=WARN \
    DISABLE_AUTOUPDATER=1

RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      ca-certificates curl git jq rsync tmux && \
    rm -rf /var/lib/apt/lists/*

RUN git clone "$NANOGPT_REPOSITORY" /opt/nanogpt && \
    git -C /opt/nanogpt checkout --detach "$NANOGPT_REVISION" && \
    test "$(git -C /opt/nanogpt rev-parse HEAD)" = "$NANOGPT_REVISION" && \
    test -z "$(git -C /opt/nanogpt status --short)"

RUN curl --fail --location --silent --show-error \
      "https://github.com/openai/codex/releases/download/rust-v${CODEX_VERSION}/codex-package-x86_64-unknown-linux-musl.tar.gz" \
      --output /tmp/codex-package.tar.gz && \
    printf '%s  %s\n' "$CODEX_SHA256" /tmp/codex-package.tar.gz | sha256sum --check --strict - && \
    mkdir -p /opt/codex && \
    tar -xzf /tmp/codex-package.tar.gz -C /opt/codex && \
    ln -s /opt/codex/bin/codex /usr/local/bin/codex && \
    ln -s /opt/codex/bin/codex-code-mode-host /usr/local/bin/codex-code-mode-host && \
    rm -f /tmp/codex-package.tar.gz && \
    codex-code-mode-host --help >/dev/null && \
    codex --version

RUN curl --fail --location --silent --show-error \
      "https://downloads.claude.ai/claude-code-releases/${CLAUDE_VERSION}/linux-x64/claude" \
      --output /tmp/claude && \
    printf '%s  %s\n' "$CLAUDE_SHA256" /tmp/claude | sha256sum --check --strict - && \
    install -m 0755 /tmp/claude /usr/local/bin/claude && \
    rm -f /tmp/claude && \
    claude --version

COPY requirements.txt /opt/dmpod/requirements.txt
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple \
      -r /opt/dmpod/requirements.txt && \
    python -m pip check

COPY dmpod/ /opt/dmpod/
COPY DMPOD_VERSION /opt/dmpod/VERSION
COPY scripts/container-entrypoint.sh /usr/local/bin/dmpod-entrypoint
RUN test -n "${DMPOD_VERSION}" && \
    test "$(tr -d '[:space:]' < /opt/dmpod/VERSION)" = "${DMPOD_VERSION}" && \
    chmod 0755 /usr/local/bin/dmpod-entrypoint /opt/dmpod/bin/* && \
    git -C /opt/nanogpt apply --check /opt/dmpod/patches/nanogpt-atomic-checkpoints.patch && \
    for command in /opt/dmpod/bin/*; do \
      ln -s "$command" "/usr/local/bin/$(basename "$command")"; \
    done && \
    sed -i \
      -e '/^cat \/etc\/runpod\.txt$/d' \
      -e '/For detailed documentation and guides/d' \
      /root/.bashrc && \
    printf '\nsource /opt/dmpod/shell-banner.sh\n' >> /root/.bashrc && \
    python -m compileall -q /opt/dmpod

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/dmpod-entrypoint"]
CMD ["/start.sh"]

FROM runtime AS test
COPY tests/ /tmp/dmpod-source/tests/
COPY dmpod/ /tmp/dmpod-source/dmpod/
COPY scripts/ /tmp/dmpod-source/scripts/
COPY runpod/ /tmp/dmpod-source/runpod/
COPY DMPOD_VERSION /tmp/dmpod-source/DMPOD_VERSION
RUN cd /tmp/dmpod-source && python -m unittest discover -s tests -v && \
    DMPOD_WORKSPACE=/tmp/dmpod-smoke-workspace \
      dmpod-entrypoint /bin/bash -c \
      'test -x /start.sh && test -d .git && test -f AGENTS.md && dmpod-setup --wandb-mode offline --skip-hf --non-interactive' && \
    rm -rf /tmp/dmpod-source /tmp/dmpod-smoke-workspace

FROM runtime AS final
