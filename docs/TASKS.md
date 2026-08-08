# Tasks

Tasks are model-agnostic experiment specifications. They own the family,
Dataset view, selected scenes/cases, training and inference seeds, evaluation
protocol and reporting policy. They never select a baseline conditioning mode.

The near-release provides two official Task files:

- `tasks/official/five_scene_direct_eval.json`
- `tasks/official/five_scene_finetune_eval.json`

Both target Dataset 13.0.0 and protocol `scene_default_v10`. Both select the
same five scored scenes. The two preview scenes are intentionally absent.

Compile a Task for one baseline without running it:

```bash
physbench task-build \
  --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline baselines/my_model \
  --output task_instance.json
```

Run the same Task over multiple user baselines:

```bash
physbench matrix-run \
  --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/official/five_scene_direct_eval.json \
  --baseline baselines/model_a \
  --baseline baselines/model_b \
  --matrix-id comparison \
  --output-root run
```

Each matrix element remains an independent AtomicRun with its own frozen
baseline identity. A planned dry run is not a completed benchmark result.
