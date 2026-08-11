# WAN2.2 Symbol-Value Cross-Attention

This baseline fine-tunes rank-32 WAN2.2-TI2V-5B DiT LoRA together with a
symbol-value conditioner. The Dataset caption remains unchanged. Frozen UMT5
subword embeddings represent each audited physical symbol; SI value, dimension,
and unit embeddings are fused with those symbol vectors and injected into the
frozen text context through trainable bottleneck cross-attention.

The portable manifest fixes the model and training semantics. Copy
`baseline.local.example.json` to the ignored `baseline.local.json` and configure
only the machine-local Python, model, project, accelerator, and GPU paths.

Validate the registration with:

```bash
PYTHONPATH=src python -m physbench baseline validate \
  wan22_ti2v_5b_lora_r32_symbol_value_cross_attention_v1
```
