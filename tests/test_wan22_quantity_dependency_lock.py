from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from physbench.baselines.wan22_quantity import (
    Wan22QuantityLoraAdapter,
)
from physbench.io import load_json


EXPECTED_DIFFSYNTH_COMMIT = (
    "fb337fbb90945ff829de69dbd44ded618f73e889"
)


def _adapter(
    runtime_root: Path,
    accelerate_config: str | None,
    *,
    diffsynth_commit: str | None = EXPECTED_DIFFSYNTH_COMMIT,
) -> Wan22QuantityLoraAdapter:
    runtime = {
        "project_root": str(runtime_root),
        "python": "python",
        "model_base": "models",
        "diffsynth_commit": diffsynth_commit,
    }
    if accelerate_config is not None:
        runtime["accelerate_config"] = accelerate_config
    return Wan22QuantityLoraAdapter({
        "baseline_id": "quantity-test",
        "input_view": "i2v",
        "runtime": runtime,
        "media_adapter": {
            "width": 480,
            "height": 832,
            "fps": 24,
            "max_frames": 121,
            "min_frames": 5,
        },
        "quantity_encoder": {"text_hidden_size": 16},
        "lora": {},
    })


class Wan22QuantityDependencyLockTests(unittest.TestCase):
    def test_training_environment_uses_audited_run_local_config(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_root = root / "runtime"
            source = runtime_root / "configs" / "accelerate.yaml"
            source.parent.mkdir(parents=True)
            original = (
                "compute_environment: LOCAL_MACHINE\n"
                "distributed_type: MULTI_GPU\n"
                "num_processes: 8\n"
            ).encode()
            source.write_bytes(original)
            adapter = _adapter(
                runtime_root,
                "configs/accelerate.yaml",
            )
            run_dir = root / "run"

            environment = adapter._training_environment(
                run_dir,
                run_dir / "dataset",
                run_dir / "dataset" / "metadata.jsonl",
            )

            frozen = (
                run_dir
                / "artifacts"
                / "wan22"
                / "accelerate_config.yaml"
            ).resolve()
            audit_path = (
                run_dir
                / "artifacts"
                / "wan22"
                / "accelerate_config.audit.json"
            )
            digest = hashlib.sha256(original).hexdigest()
            self.assertEqual(str(frozen), environment["ACCELERATE_CONFIG"])
            self.assertEqual(
                digest,
                environment["ACCELERATE_CONFIG_SHA256"],
            )
            self.assertEqual(original, frozen.read_bytes())
            audit = load_json(audit_path)
            self.assertEqual("1.0", audit["schema_version"])
            self.assertEqual(str(source.resolve()), audit["source_path"])
            self.assertEqual(str(frozen), audit["frozen_path"])
            self.assertEqual(digest, audit["source_sha256"])
            self.assertEqual(digest, audit["frozen_sha256"])
            self.assertEqual(len(original), audit["size"])

            source.write_text(
                "distributed_type: NO\nnum_processes: 1\n",
                encoding="utf-8",
            )
            source.unlink()
            repeated = adapter._training_environment(
                run_dir,
                run_dir / "dataset",
                run_dir / "dataset" / "metadata.jsonl",
            )
            self.assertEqual(digest, repeated["ACCELERATE_CONFIG_SHA256"])
            self.assertEqual(original, frozen.read_bytes())

    def test_existing_frozen_config_is_verified_before_reuse(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "runtime" / "accelerate.yaml"
            source.parent.mkdir(parents=True)
            source.write_text("num_processes: 8\n", encoding="utf-8")
            adapter = _adapter(source.parent, source.name)
            run_dir = root / "run"
            adapter._freeze_accelerate_config(run_dir)
            frozen = adapter._accelerate_config_path(run_dir)
            frozen.write_text("num_processes: 1\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "differs from its recorded SHA-256",
            ):
                adapter._freeze_accelerate_config(run_dir)

    def test_accelerate_config_is_required_and_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            planning_adapter = _adapter(root, None)
            planned_environment = planning_adapter._training_environment(
                root / "planned-run",
                root / "dataset",
                root / "dataset" / "metadata.jsonl",
            )
            self.assertNotIn("ACCELERATE_CONFIG", planned_environment)

            with self.assertRaisesRegex(
                ValueError,
                "runtime.accelerate_config is required",
            ):
                planning_adapter._freeze_accelerate_config(root / "run")
            planning_adapter.execute = True
            with self.assertRaisesRegex(
                ValueError,
                "runtime.accelerate_config is required",
            ):
                planning_adapter._training_environment(
                    root / "execute-run",
                    root / "dataset",
                    root / "dataset" / "metadata.jsonl",
                )
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Accelerate config not found",
            ):
                _adapter(
                    root,
                    "missing.yaml",
                )._freeze_accelerate_config(root / "run")

    def test_diffsynth_commit_pin_is_mandatory_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid_pins = (
                None,
                "",
                EXPECTED_DIFFSYNTH_COMMIT[:12],
                EXPECTED_DIFFSYNTH_COMMIT.upper(),
            )
            for pin in invalid_pins:
                with self.subTest(pin=pin):
                    adapter = _adapter(
                        root,
                        "accelerate.yaml",
                        diffsynth_commit=pin,
                    )
                    with (
                        patch(
                            "physbench.baselines.wan22_quantity."
                            "subprocess.check_output"
                        ) as check_output,
                        self.assertRaisesRegex(
                            ValueError,
                            "must pin an exact lowercase",
                        ),
                    ):
                        adapter._verified_diffsynth_commit(root)
                    check_output.assert_not_called()

            adapter = _adapter(root, "accelerate.yaml")
            with patch(
                "physbench.baselines.wan22_quantity."
                "subprocess.check_output",
                return_value=EXPECTED_DIFFSYNTH_COMMIT + "\n",
            ) as check_output:
                actual = adapter._verified_diffsynth_commit(root)
            self.assertEqual(EXPECTED_DIFFSYNTH_COMMIT, actual)
            command = check_output.call_args.args[0]
            self.assertEqual("rev-parse", command[-3])
            self.assertEqual("--verify", command[-2])
            self.assertEqual("HEAD^{commit}", command[-1])

            different = "0" * 40
            with (
                patch(
                    "physbench.baselines.wan22_quantity."
                    "subprocess.check_output",
                    return_value=different + "\n",
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "deployment commit differs",
                ),
            ):
                adapter._verified_diffsynth_commit(root)


if __name__ == "__main__":
    unittest.main()
