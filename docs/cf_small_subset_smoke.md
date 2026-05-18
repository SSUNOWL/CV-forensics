# CF-Small Local Subset Smoke Plan

<!-- CF_SMALL_SUBSET_SMOKE_OK -->

This document defines a planning and validation step for a tiny Community
Forensics-Small local subset smoke workflow. It is not full training.

## Scope

This smoke plan is symbolic and local-path gated. It does not download a
dataset, read real images, read real masks, inspect dataset directories
recursively, write prediction artifacts, or write checkpoint artifacts.

The example config lives at
`configs/local_data/cf_small_subset_smoke.example.json` and keeps all path
references symbolic until explicit user approval is given.

## Local-Path Approval Gate

Real local paths are not configured in the example file. A real local subset
smoke run can only happen after:

- explicit user approval for local data access
- the task 0011 local data readiness gate passes
- the CF-Small manifest is validated
- approved local paths are provided through a later approved config
- any later artifact policy is explicitly decided before artifact writing

Until those conditions are met, validation stays at config structure and
manifest-reference planning only.

## Tiny Subset Plan

The symbolic subset plan limits the planned smoke to a small number of manifest
rows. It checks only that required metadata fields are planned:

- `architecture`
- `model_name`
- `subset`

The planned smoke does not enumerate dataset files and does not open media.

## Holdout Planning

CF-Small is important because unseen generator generalization is central to the
project. For that reason, a generator-holdout or model-name-holdout plan is
required before using the subset for any meaningful validation.

Random-only validation is insufficient because it can place samples from the
same generator family or model name in both train and validation subsets, which
would overstate generalization.

## Next-Step Policy

When real local paths are explicitly approved, the next step is still a small
local subset smoke run, not full training. The run should validate manifest row
loading and split planning first, then stop before reading images, writing
artifacts, or launching training unless a later task explicitly approves that
larger scope.

SNS augmentation is not implemented here.
