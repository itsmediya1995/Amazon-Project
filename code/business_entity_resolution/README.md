# AetherLink — Business Entity Resolution

Parse → retrieve → rank → assign pipeline. Country-open (US / India / France).
No Kaggle notebook code.

## Run (from `student_resource/`)

```bash
python code/business_entity_resolution/src/run.py --data-root dataset --out-dir output
```

Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
