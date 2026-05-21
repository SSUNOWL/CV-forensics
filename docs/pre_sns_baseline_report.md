# Pre-SNS Baseline Report

`PRE_SNS_BASELINE_REPORT_OK`

This document describes the pre-SNS baseline report generator. The generator
collects already-created JSON results from pilot training, single-image
inference, pre-SNS evaluation, and scaled training, then summarizes them before
SNS augmentation begins.

## Expected Inputs

The config must list explicit local JSON result paths:

- `training_result_path`
- `inference_report_path`
- `evaluation_result_path`
- `scaled_training_result_path`

The generator reads those paths only. It does not scan directories, train, run
inference, run evaluation, download data, write checkpoints, or apply SNS
augmentation.

## Generated Artifacts

When `write_report=true`, the generator writes a small Markdown report and JSON
summary under `approved_report_root`, which must be outside the repository and
not under repository `outputs/` or `checkpoints/`.

The tracked document is a template and policy note only. It must not contain
private local paths from a lab run.

## No-SNS Status

The report must clearly state that SNS augmentation has not been applied yet.
It is a pre-SNS baseline snapshot used to decide the next SNS augmentation and
robustness-evaluation step.

## Claim Boundary

The report is not a final full-dataset performance claim unless scaled or full
evaluation has been separately approved, completed, and documented.
