# Pre-SNS Training Preflight Dry-Run

This document defines the final dry-run gate before a guarded training entrypoint.

The preflight consumes the unified pre-SNS manifest schema, checks class coverage, family/provenance coverage, tampered mask coverage, CPU-only limits, max sample limits, and loss routing. In approved local mode it may read only explicitly listed temporary or user-approved image and mask paths from an approved manifest.

The runner performs one tiny CPU-only forward/backward pass through the integrated model to verify model wiring and finite class, family, localization, and total losses. It does not run a real training loop and does not run an optimizer step by default.

No outputs are written. No checkpoints are written. No SNS augmentation is implemented. No SNS evaluation is run. This is not a performance claim.

Marker: PRE_SNS_TRAINING_PREFLIGHT_OK
