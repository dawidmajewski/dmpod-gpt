---
name: dmpod-huggingface-publish
description: Prepare, review, and optionally publish a trained DMPod nanoGPT run to a Hugging Face model repository. Use for model cards, README generation, benchmark tables, W&B images, checkpoints, tokenizers, hf upload, or any request to share a trained model on Hugging Face.
---

# Publish A DMPod Model To Hugging Face

1. Read the run's `state.json`, `summary.json`, `artifacts.json`, `wandb.json`,
   and benchmark results. Do not present an incomplete or limited benchmark run
   as a completed public result.
2. If full quality benchmarks are absent, ask whether the user wants them. Do
   not start them without approval because they download data and consume GPU
   time. If approved, run `dmpod-benchmark NAME`.
3. Generate the publication files with `dmpod-export-run NAME`. Pass the user-
   approved `--model-license`, model ID, languages, dataset IDs, and tags when
   they are not already recorded by the profile.
4. To include W&B screenshots or report images, download them to a directory
   under `/workspace` and pass `--wandb-media-dir PATH`. The exporter generates
   scalar training curves directly from `metrics.jsonl`; W&B UI charts are not
   assumed to be downloadable run files.
5. Review `reports/README.md`, `reports/report-context.json`, generated assets,
   benchmark limitations, intended use, risks, source hashes, and license. The
   output is a starting model card and must remain editable by the user.
6. Describe the artifact as a native trusted nanoGPT training checkpoint unless a
   verified Transformers conversion is also provided. Never claim
   `AutoModel.from_pretrained` compatibility for the native checkpoint.
7. The default reviewed file set is the generated `README.md`,
   `report-context.json`, `metrics.jsonl`, `assets/`, full benchmark results,
   `config.json`, `sources/model.py`, `sources/tokenizer.json`, and the selected
   final checkpoint renamed to `checkpoint.pt`. Include other source snapshots
   when they help reproducibility; exclude optimizer-only or private files that
   are not intended for the public repository.
8. Before upload, show the exact repository ID, source-to-destination file map,
   and total size, then obtain explicit approval. Never expose `HF_TOKEN`. Use
   the installed `hf` CLI and inspect `hf upload --help` for the image's pinned
   CLI version before constructing the upload command.
9. After upload, verify the public model card, files, checkpoint checksum,
   tokenizer checksum, W&B link visibility, images, and benchmark tables from
   the Hugging Face repository page.
