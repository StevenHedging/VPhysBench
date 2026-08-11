# Exclude Push Bottle from Task v1

## Goal

Remove the `push_bottle` scene from every official Task v1 training and
evaluation workload without deleting or republishing any Dataset content.

## Decision

The only finetune Task is corrected in place from
`seven_scene_train_five_scene_eval_v1` to
`six_scene_train_five_scene_eval_v1`. Its `training_scene_ids` selector drops
`push_bottle`; its existing five-scene evaluation selector is unchanged because
it already excludes that scene.

This keeps Task v1 as the single current protocol rather than creating a new
version. Renaming the Task and file is required so the public identity remains
truthful. Keeping the old seven-scene name would make canonical plans and result
provenance misleading.

## Resulting contract

- Finetune training: six scenes and 679 View A train cases.
- Finetune evaluation: the existing five scored scenes and 76 View A ID test
  cases.
- Direct evaluation: unchanged at five scenes and 658 cases.
- Dataset: unchanged at seven scenes and 916 cases.
- `push_bottle`: all 127 train and 14 test cases remain in the Dataset, assets,
  Distribution v1 shards, and immutable Hugging Face revision, but no official
  Task v1 selects them.
- Evaluation protocol and Dataset identity: unchanged.

## Change boundary

Change the official Task JSON, release manifest, public documentation, and
references in contract tests. Do not change the Task schema, planner, evaluator,
Dataset descriptors, distribution manifest, shard contents, Hub binding, or
baseline capabilities.

Historical design records remain historical and are not rewritten.

## Verification

Contract tests must fail before the Task declaration changes and then prove:

1. the official Task directory exposes only the five-scene direct Task and the
   renamed six-scene finetune Task;
2. the finetune canonical plan has exactly six training scenes and 679 training
   cases;
3. neither its training cases nor evaluation jobs contain `push_bottle`;
4. the Dataset still contains the `push_bottle` scene with 127 train and 14 test
   cases; and
5. interface, release audit, and release archive checks remain green.
