# WAN2.2 Symbol–Value Cross-Attention Baseline

Baseline ID:
`wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1`.

该 Baseline 使用 Dataset 原始 caption 作为 T₀，不追加任何数值，也不使用 sentinel
替换。Bundle-local registry 从 `case.physics` 选择独立标量，并严格校验每个 symbol
确实出现在 T₀ 中。

## 条件编码

```text
T₀ ── frozen UMT5 self-attention ─────────────── H_text [L,4096]

symbol ── frozen UMT5 subword embedding mean ── E_symbol [m,4096]
value/unit ── SI features + dimension + unit ── E_value  [m,4096]
                         LayerNorm(E_symbol + E_value)
                                      │
H_text query ── 512-d / 8-head cross-attention ◄┘
                                      │
                         gated residual text context
                                      │
                                  WAN DiT
```

UMT5、VAE 与 WAN base 权重冻结；仅训练 conditioner 和 rank-32 DiT LoRA。负面
CFG 分支不注入物理 token。Combined safetensors 在加载 LoRA 或 conditioner 之前会
校验 manifest hash、conditioner 完整 topology、600 个 LoRA tensor 及目标层集合。

## 训练配置

- Dataset 13 View A 的 7 个 scene；train 806 cases，test 110 cases；
- 每个 scene 确定性 oversample 到 312 行，总计 2184 行/epoch；
- 8×A100 40GB，bf16 WAN、FP32 conditioner；
- 2 epochs，546 optimizer steps，AdamW，LR `1e-4`，weight decay `0.01`；
- seed 42，121 frames，24 fps；推理 50 steps，CFG 5.0。

`push_bottle` 和 `vertical_spring_oscillator` 当前没有 evaluator。实验协议把二者显式
标成 `unsupported`：仍训练、生成并保存全部 run 产物，但不生成虚构分数。

## 命令

```bash
PYTHONPATH=src /root/miniconda3/envs/dlp/bin/python -m physbench \
  baseline validate \
  wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1

PYTHONPATH=src /root/miniconda3/envs/dlp/bin/python -m physbench \
  task-build \
  --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/experiments/seven_scene_symbol_value_finetune_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1 \
  --output results/seven_scene_symbol_value_task_instance.json

PYTHONPATH=src /root/miniconda3/envs/dlp/bin/python -m physbench \
  atomic-run \
  --dataset datasets/releases/13.0.0/dataset.json \
  --task tasks/experiments/seven_scene_symbol_value_finetune_eval.json \
  --baseline wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1 \
  --run-id wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1 \
  --output-root run \
  --execute
```

AtomicRun 会保留 frozen identities、806 个训练 adaptation、110 个 inference job、
采样计划、模型资产 hash、训练 loss/gradient、combined checkpoint、逐 case token
attention audit、预测视频、支持 scene 的 evaluator 输出以及 unsupported scene 记录。

