# Pre-SNS Baseline Evaluation Report Scaffold

<!-- PRE_SNS_BASELINE_REPORT_OK -->

This document defines the pre-SNS baseline evaluation report scaffold. It is the
comparison anchor for later SNS robustness analysis and SNS augmentation-aware
training comparisons.

This is not dataset download, not training, not real evaluation, and not real
image or mask reading. No real metric values are claimed in this task; every
metric value remains pending until explicit user approval for a real baseline
evaluation.

## Purpose

The scaffold records which baseline metrics must exist before any SNS stage
begins. Later SNS perturbation robustness work can compare against this anchor,
including future robustness drop fields and future SNS-augmented training
comparisons.

## Dependencies

This report scaffold depends on:

- task 0007 model output schema
- task 0008 fake inference
- task 0009 metrics
- task 0010 training dry-run
- task 0011 local data readiness
- task 0012 CF-Small subset smoke
- task 0013 SID-Set subset smoke
- task 0014 CF-Small baseline plan
- task 0015 SID-Set multi-head baseline plan

## Required Metric Fields

The pre-SNS baseline report must reserve placeholders for:

- 3-way accuracy
- Macro-F1
- mask IoU
- generator-family accuracy
- localization activation recall
- latency
- FPS

Every metric value stays null with status `pending_real_baseline_run` until the
user explicitly approves real baseline evaluation.

## SNS Boundary

SNS robustness drop is a future comparison field and remains pending until
SNS-stage tasks. SNS augmentation is not implemented here, and SNS perturbation
evaluation is not run here.

## Before Real Baseline Evaluation

Before any real pre-SNS baseline evaluation:

- explicit user approval is required
- local path policy must be validated
- manifests must be validated
- local subset smoke confirmation must pass
- output policy must be approved
- checkpoint policy must be approved
- compute budget policy must be approved
