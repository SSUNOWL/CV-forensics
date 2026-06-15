# SIDA Clean-vs-SNSAug Paired Diagnostic

SNSAUG_V2_SIDA_CLEAN_VS_SNSAUG_OK

Task 0079 exports clean/basic counterparts for the SNSAug SIDA subset and evaluates cached clean SIDA outputs against cached SNSAug SIDA outputs.

The diagnostic has two modes:

- `export_clean_counterpart_mode`
- `cached_clean_vs_snsaug_eval_mode`

It keeps synthetic samples in 3-way classification, excludes synthetic samples from tampered mask IoU, and reports synthetic mask false-positive rates separately. Tampered samples report clean IoU, SNS IoU, delta IoU, and mask missing.

The report separates clean reference, Type A local overlay, Type B global geometry/degradation, and strict Type A+B summaries.
