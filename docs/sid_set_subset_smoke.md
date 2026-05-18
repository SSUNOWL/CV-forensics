# SID-Set Local Subset Smoke Plan

<!-- SID_SET_SUBSET_SMOKE_OK -->

This document defines a symbolic local subset smoke plan for SID-Set. The goal
is to prepare a tiny, user-approved local-path workflow for 3-way
classification and tampered localization planning.

## Scope

This is not a dataset download, not training, and not real image or mask
reading. The plan does not inspect real dataset directories, write prediction
artifacts, or write checkpoints. It only validates symbolic configuration
structure and the policies that must be in place before any real local smoke
run.

## Project Mapping

SID-Set maps to the project as the planning source for:

- 3-way classification: `real`, `synthetic`, `tampered`
- tampered localization planning
- mask IoU as the planned localization metric
- localization activation recall for the conditional localization gate
- family/provenance policy when labels are missing or coarse

Localization is primarily relevant for tampered samples. Real samples should
not require tampered masks, and synthetic samples may omit tampered masks unless
they are explicitly annotated.

## Readiness Dependency

This plan depends on the task 0011 local data readiness gate. Real local paths
must remain unavailable until explicit user approval, validated local path
policy, and manifest validation are complete.

## Difference From CF-Small Smoke

The task 0012 CF-Small smoke plan focuses on backbone and provenance planning
with generator/model-name holdout concerns. This SID-Set plan focuses on
3-way classification, tampered mask policy, conditional localization, mask IoU,
and localization activation recall.

Because SID-Set may not provide the same coarse family labels as CF-Small,
missing family/provenance labels are allowed during planning. Real samples map
to `Real-or-N/A`; synthetic and tampered samples map to `Unknown` until labels
are confirmed. Provenance head training remains gated.

## Before Real SID-Set Smoke Execution

Before any real SID-Set local subset smoke run:

- explicit user approval is required
- local path policy must be validated
- the SID-Set manifest must be validated
- output/checkpoint policy must be explicitly approved if artifacts are needed
- tiny subset size must be confirmed

SNS augmentation is not implemented here.
