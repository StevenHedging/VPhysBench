# Baseline integrations

This directory is intentionally empty in a clean VPhysBench checkout. It is
the extension root for algorithms integrated by benchmark users; VPhysBench
does not ship a built-in model baseline.

Create an image-to-video integration with:

```bash
physbench baseline init my_model --backend managed-i2v
```

The command creates `baselines/my_model/`. Commit the portable descriptor,
driver, and example configuration when sharing an integration. Keep
`baseline.local.json`, credentials, weights, checkpoints, caches, and
machine-local paths outside Git.

See `docs/CUSTOM_BASELINE_QUICKSTART.md` for the command contract and the
first one-case run.
