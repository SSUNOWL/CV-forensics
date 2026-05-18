# SID-Set Multi-Head Baseline Plan

<!-- SID_SET_BASELINE_PLAN_OK -->

This document defines the plan-only SID-Set multi-head fine-tuning and
localization baseline scaffold. It prepares the repository for a later
explicitly approved baseline, but it does not download SID-Set, run training,
read real images, read real masks, write outputs, or write checkpoints.

## Project Mapping

SID-Set maps to the project goals as the planning source for:

- 3-way classification: `real`, `synthetic`, `tampered`
- tampered localization
- mask IoU
- localization activation recall
- family/provenance handling when labels are missing or coarse
- deterministic reason/template explanation

## Planned Architecture

The planned multi-head structure is:

- shared visual backbone
- 3-way classification head
- generator-family provenance head
- conditional localization head
- evidence aggregation
- deterministic template explanation

CF-Small can contribute backbone and provenance initialization. SID-Set handles
3-way classification and tampered localization planning.

## Dependencies

This plan depends on:

- the task 0011 local data readiness gate
- the task 0012 CF-Small subset smoke plan
- the task 0013 SID-Set subset smoke plan
- the task 0014 CF-Small baseline plan

## Before Real Fine-Tuning

Before any real SID-Set baseline fine-tuning:

- explicit user approval is required
- local path policy must be validated
- the SID-Set manifest must be validated
- tiny subset smoke confirmation must pass
- output policy must be approved
- checkpoint policy must be approved
- compute budget policy must be approved

## Planned Metrics

The planned metrics are:

- 3-way accuracy
- Macro-F1
- mask IoU
- generator-family accuracy
- localization activation recall
- latency
- FPS

SNS augmentation is not implemented here. Task 0016 will collect pre-SNS
baseline evaluation report scaffolding before any SNS augmentation work begins.
