# CF-Small Baseline Training Plan

<!-- CF_SMALL_BASELINE_PLAN_OK -->

This document defines the plan-only Community Forensics-Small baseline
training/evaluation scaffold. It prepares the repository for a later explicitly
approved baseline, but it does not download a dataset, run training, read real
images, read masks, write artifacts, or write checkpoints.

## Project Mapping

Community Forensics-Small maps to this project as the planning source for:

- shared visual backbone planning
- real-vs-synthetic baseline planning
- generator-family/provenance planning
- generator/model-family holdout planning

The baseline starts with binary real-vs-synthetic discrimination and coarse
family provenance. It is not the primary tampered localization source.

## Localization Boundary

Community Forensics-Small is classification-centered for this project. It
should not be treated as the main tampered mask localization dataset. Mask IoU
and localization activation recall are SID-Set/localization-stage metrics, not
required CF-Small baseline metrics.

## Dependencies

This plan depends on:

- the task 0011 local data readiness gate
- the task 0012 CF-Small local subset smoke plan
- the task 0013 SID-Set local subset smoke plan

The CF-Small subset smoke plan gates local path readiness for CF-Small. The
SID-Set subset smoke plan defines the later 3-way classification and tampered
localization responsibilities.

## Before Real Baseline Training

Before any real Community Forensics-Small baseline training:

- explicit user approval is required
- local path policy must be validated
- the CF-Small manifest must be validated
- tiny subset smoke confirmation must pass
- output policy must be approved
- checkpoint policy must be approved
- compute budget policy must be approved

## Planned Metrics

The planned CF-Small baseline metrics are:

- binary accuracy
- macro F1
- generator-family accuracy
- holdout/generalization evaluation
- latency
- FPS

SNS augmentation is not implemented here.
