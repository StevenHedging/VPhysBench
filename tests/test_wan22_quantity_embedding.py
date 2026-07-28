from __future__ import annotations

import copy
import json
import math
import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from _paths import ROOT
from physbench.baseline_api import (
    load_baseline_bundle,
    load_baseline_plugin,
)
from physbench.baseline_runtime.input_contract import (
    MAX_INLINE_CONTROL_BYTES,
)
from physbench.baselines.wan22_media import Wan22MediaAdapter
from physbench.baselines.wan22_quantity import (
    Wan22QuantityLoraAdapter,
)
from physbench.baselines.wan22_quantity_model import (
    NUMERIC_FEATURE_NAMES,
    QUANTITY_CHECKPOINT_PREFIX,
    QuantityEncoder,
    install_quantity_prompt_unit,
    load_combined_quantity_checkpoint,
    load_quantity_encoder_checkpoint,
    locate_sentinel_tokens,
    numeric_features,
    quantity_inference_conditioning,
    quantity_state_dict,
)
from physbench.data_layout import V4_DATASET
from physbench.datasets import load_dataset
from physbench.domain import TaskSpec
from physbench.io import (
    canonical_sha256,
    load_json,
    load_jsonl,
)
from physbench.tasks import load_task


BASELINE = (
    ROOT / "baselines" / "wan22_quantity_embedding" / "baseline.json"
)
FINETUNE_TASK = (
    ROOT / "tasks" / "official" / "five_scene_finetune_eval.json"
)

QUANTITY_FIELDS = {
    "name",
    "raw_value",
    "raw_unit",
    "rendered_value",
    "rendered_quantity",
    "si_value",
    "canonical_si_unit",
    "dimension",
    "dimension_basis",
    "quantity_type",
    "quantity_type_id",
    "source_role",
    "sentinel",
    "audited_char_span",
    "model_char_span",
}

EXPECTED_UNITS = {
    "1": ("1", 1.0, [0, 0, 0, 0, 0, 0, 0]),
    "m": ("m", 1.0, [1, 0, 0, 0, 0, 0, 0]),
    "kg": ("kg", 1.0, [0, 1, 0, 0, 0, 0, 0]),
    "m/s": ("m/s", 1.0, [1, 0, -1, 0, 0, 0, 0]),
    "m/s^2": ("m/s^2", 1.0, [1, 0, -2, 0, 0, 0, 0]),
    "N": ("kg*m/s^2", 1.0, [1, 1, -2, 0, 0, 0, 0]),
    "deg": (
        "rad",
        0.017453292519943295,
        [0, 0, 0, 0, 0, 0, 0],
    ),
    "deg/s": (
        "rad/s",
        0.017453292519943295,
        [0, 0, -1, 0, 0, 0, 0],
    ),
}


def _create_run_directories(run_dir: Path) -> None:
    for child in (
        "adaptations",
        "artifacts",
        "jobs",
        "logs",
        "predictions",
        "provenance",
        "task_instance",
        "training",
    ):
        (run_dir / child).mkdir(parents=True)


def _small_encoder_config() -> dict:
    return {
        "numeric_feature_size": len(NUMERIC_FEATURE_NAMES),
        "numeric_hidden_size": 8,
        "numeric_embedding_size": 6,
        "dimension_size": 7,
        "dimension_hidden_size": 6,
        "dimension_embedding_size": 5,
        "quantity_type_count": 10,
        "quantity_type_embedding_size": 4,
        "fusion_hidden_size": 12,
        "text_hidden_size": 9,
    }


def _fake_safetensors_modules(
    state: dict[str, torch.Tensor],
) -> dict[str, ModuleType]:
    package = ModuleType("safetensors")
    torch_module = ModuleType("safetensors.torch")
    torch_module.load_file = lambda _path, device="cpu": {  # type: ignore[attr-defined]
        key: value.to(device)
        for key, value in state.items()
    }
    package.torch = torch_module  # type: ignore[attr-defined]
    return {
        "safetensors": package,
        "safetensors.torch": torch_module,
    }


def _write_safetensors_header(
    path: Path,
    tensors: dict[str, tuple[str, list[int]]],
) -> None:
    offset = 0
    header = {}
    dtype_sizes = {"F32": 4, "BF16": 2}
    for name, (dtype, shape) in tensors.items():
        count = math.prod(shape)
        size = count * dtype_sizes[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    payload = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


class _FakeTokenizerBackend:
    unk_token_id = 0

    @staticmethod
    def convert_tokens_to_ids(token: str) -> int:
        match = re.fullmatch(r"<extra_id_(\d+)>", token)
        return 1000 + int(match.group(1)) if match else 0


class _FakeWanTokenizer:
    clean = "whitespace"

    def __init__(self):
        self.tokenizer = _FakeTokenizerBackend()

    @staticmethod
    def _clean(text: str) -> str:
        return text

    def __call__(
        self,
        prompt: str,
        *,
        return_mask: bool,
        add_special_tokens: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del return_mask, add_special_tokens
        pieces = re.split(r"(<extra_id_\d+>)", prompt)
        token_ids = [11]
        for piece in pieces:
            if not piece:
                continue
            if re.fullmatch(r"<extra_id_\d+>", piece):
                token_ids.append(
                    self.tokenizer.convert_tokens_to_ids(piece)
                )
            else:
                token_ids.append(12)
        token_ids.append(1)
        ids = torch.tensor([token_ids], dtype=torch.long)
        return ids, torch.ones_like(ids)


class _FakePipelineUnit:
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.seperate_cfg = kwargs.get("seperate_cfg", False)
        self.take_over = kwargs.get("take_over", False)


def _fake_diffsynth_pipeline_modules() -> dict[str, ModuleType]:
    package = ModuleType("diffsynth")
    diffusion = ModuleType("diffsynth.diffusion")
    base_pipeline = ModuleType("diffsynth.diffusion.base_pipeline")
    base_pipeline.PipelineUnit = _FakePipelineUnit  # type: ignore[attr-defined]
    package.diffusion = diffusion  # type: ignore[attr-defined]
    diffusion.base_pipeline = base_pipeline  # type: ignore[attr-defined]
    return {
        "diffsynth": package,
        "diffsynth.diffusion": diffusion,
        "diffsynth.diffusion.base_pipeline": base_pipeline,
    }


class _FakeDit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.proj = nn.Linear(3, 4, bias=False)


class _FakeLoraPipe:
    def __init__(self) -> None:
        self.dit = _FakeDit()
        self.load_lora_calls = 0
        self.loaded_lora_keys: set[str] = set()

    def load_lora(
        self,
        module: nn.Module,
        *,
        state_dict: dict[str, torch.Tensor],
        alpha: float,
    ) -> None:
        self.load_lora_calls += 1
        self.loaded_lora_keys = set(state_dict)
        modules = dict(module.named_modules())
        with torch.no_grad():
            for key, tensor_b in state_dict.items():
                if not key.endswith(".lora_B.default.weight"):
                    continue
                target = key.removesuffix(".lora_B.default.weight")
                tensor_a = state_dict[
                    f"{target}.lora_A.default.weight"
                ]
                modules[target].weight.add_(
                    float(alpha) * torch.mm(tensor_b, tensor_a)
                )


class Wan22QuantityEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(V4_DATASET, check_assets=False)
        cls.by_id = {
            case["case_id"]: case for case in cls.dataset.cases
        }
        cls.bundle = load_baseline_bundle(BASELINE)
        cls.plugin = load_baseline_plugin(cls.bundle)
        cls.adapter = cls.plugin.task_builder.data_adapter
        cls.finetune_task = load_task(FINETUNE_TASK)

    def test_all_214_cases_adapt_without_task_compilation(self) -> None:
        self.assertEqual(214, len(self.dataset.cases))
        adaptations = [
            self.adapter.adapt_case(case, role="eval")
            for case in self.dataset.cases
        ]
        self.assertEqual(214, len(adaptations))
        self.assertEqual(
            {case["case_id"] for case in self.dataset.cases},
            {item["case_id"] for item in adaptations},
        )
        self.assertTrue(all(
            adaptation["role"] == "eval"
            for adaptation in adaptations
        ))
        for adaptation in adaptations:
            self.assertEqual(
                "quantity_token_embedding_v1",
                adaptation["native_inputs"]["physics"]["representation"],
            )
            self.assertTrue(
                adaptation["native_inputs"]["physics"]["quantities"]
            )

    def test_quantity_registry_units_fields_and_spans_are_auditable(
        self,
    ) -> None:
        registry = self.adapter.registry.value
        self.assertEqual(
            [
                "L",
                "M",
                "T",
                "I",
                "Theta",
                "N_amount",
                "J_luminous",
            ],
            registry["dimension_basis"],
        )
        self.assertEqual(set(EXPECTED_UNITS), set(registry["units"]))
        for unit, (
            canonical_unit,
            scale,
            dimension,
        ) in EXPECTED_UNITS.items():
            with self.subTest(unit=unit):
                spec = registry["units"][unit]
                self.assertEqual(canonical_unit, spec["canonical_si_unit"])
                self.assertEqual(scale, spec["si_scale"])
                self.assertEqual(dimension, spec["dimension"])

        for case in self.dataset.cases:
            with self.subTest(case_id=case["case_id"]):
                adaptation = self.adapter.adapt_case(case, role="eval")
                native = adaptation["native_inputs"]
                self.assertEqual(
                    {"vision", "text", "generation_shape", "physics"},
                    set(native),
                )
                self.assertEqual(
                    {"prompt", "audited_prompt"},
                    set(native["text"]),
                )
                payload = native["physics"]
                self.assertEqual(
                    {
                        "representation",
                        "registry_id",
                        "registry_fingerprint",
                        "quantities",
                    },
                    set(payload),
                )
                self.assertEqual(
                    "quantity_token_embedding_v1",
                    payload["representation"],
                )
                inline_size = len(json.dumps(
                    payload["quantities"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"))
                self.assertLessEqual(
                    inline_size,
                    MAX_INLINE_CONTROL_BYTES,
                )

                audited_prompt = native["text"]["audited_prompt"]
                model_prompt = native["text"]["prompt"]
                names = []
                sentinels = []
                for quantity in payload["quantities"]:
                    self.assertEqual(QUANTITY_FIELDS, set(quantity))
                    name = quantity["name"]
                    names.append(name)
                    sentinels.append(quantity["sentinel"])
                    source = case["physics"][name]
                    self.assertIs(source["annotated"], True)
                    self.assertEqual(source["value"], quantity["raw_value"])
                    self.assertEqual(source["unit"], quantity["raw_unit"])
                    unit = registry["units"][quantity["raw_unit"]]
                    self.assertEqual(
                        unit["canonical_si_unit"],
                        quantity["canonical_si_unit"],
                    )
                    self.assertEqual(
                        unit["dimension"],
                        quantity["dimension"],
                    )
                    self.assertEqual(
                        registry["dimension_basis"],
                        quantity["dimension_basis"],
                    )
                    self.assertEqual(
                        registry["quantity_types"][
                            quantity["quantity_type"]
                        ],
                        quantity["quantity_type_id"],
                    )
                    self.assertAlmostEqual(
                        float(quantity["rendered_value"])
                        * float(unit["si_scale"]),
                        quantity["si_value"],
                        places=12,
                    )
                    audited_start, audited_end = quantity[
                        "audited_char_span"
                    ]
                    model_start, model_end = quantity[
                        "model_char_span"
                    ]
                    self.assertEqual(
                        quantity["rendered_quantity"],
                        audited_prompt[audited_start:audited_end],
                    )
                    self.assertEqual(
                        quantity["sentinel"],
                        model_prompt[model_start:model_end],
                    )
                    self.assertNotIn(
                        quantity["sentinel"],
                        audited_prompt,
                    )
                self.assertEqual(len(sentinels), len(set(sentinels)))
                self.assertEqual(set(names), set(adaptation["used_parameters"]))
                channel = adaptation["input_contract"][
                    "physics_channels"
                ]
                self.assertEqual(1, len(channel))
                self.assertEqual(
                    "native_inputs.physics.quantities",
                    channel[0]["binding"],
                )
                self.assertEqual("inline_json", channel[0]["transport"])
                self.assertEqual(sorted(names), channel[0]["used_parameters"])

    def test_sentinels_resolve_to_unique_single_token_spans(self) -> None:
        case = next(
            case
            for case in self.dataset.cases
            if case["scene_id"] == "collision_1d"
        )
        native = self.adapter.adapt_case(
            case,
            role="eval",
        )["native_inputs"]
        pipe = SimpleNamespace(tokenizer=_FakeWanTokenizer())
        ids, mask, audits = locate_sentinel_tokens(
            pipe,
            native["text"]["prompt"],
            native["physics"]["quantities"],
        )
        self.assertEqual(ids.shape, mask.shape)
        self.assertEqual(7, len(audits))
        positions = [item["token_span"][0] for item in audits]
        self.assertEqual(len(positions), len(set(positions)))
        self.assertTrue(all(
            item["token_span"][1] - item["token_span"][0] == 1
            for item in audits
        ))
        self.assertEqual(
            [item["name"] for item in native["physics"]["quantities"]],
            [item["name"] for item in audits],
        )

        duplicated = (
            native["text"]["prompt"]
            + " "
            + native["physics"]["quantities"][0]["sentinel"]
        )
        with self.assertRaisesRegex(ValueError, "occurs 2 times"):
            locate_sentinel_tokens(
                pipe,
                duplicated,
                native["physics"]["quantities"],
            )

        unknown_quantities = copy.deepcopy(
            native["physics"]["quantities"]
        )
        old = unknown_quantities[0]["sentinel"]
        unknown_quantities[0]["sentinel"] = "<not_a_t5_sentinel>"
        unknown_prompt = native["text"]["prompt"].replace(
            old,
            "<not_a_t5_sentinel>",
            1,
        )
        with self.assertRaisesRegex(ValueError, "unknown quantity sentinel"):
            locate_sentinel_tokens(
                pipe,
                unknown_prompt,
                unknown_quantities,
            )

    def test_cfg_binding_is_explicit_when_prompts_are_identical(self) -> None:
        stock_prompt_unit = type("WanVideoUnit_PromptEmbedder", (), {})
        loaded_models = []
        pipe = SimpleNamespace(
            units=[object(), object(), stock_prompt_unit()],
            load_models_to_device=lambda names: loaded_models.append(names),
        )
        quantities = [{"name": "speed"}]
        calls = []

        def fake_encode(_pipe, prompt, records):
            resolved = list(records or [])
            calls.append((prompt, resolved))
            context = torch.tensor(
                [1.0 if resolved else 0.0],
                dtype=torch.float32,
            )
            audit = [{"name": item["name"]} for item in resolved]
            return context, audit

        model_module = "physbench.baselines.wan22_quantity_model"
        with (
            patch.dict(
                sys.modules,
                _fake_diffsynth_pipeline_modules(),
            ),
            patch(
                f"{model_module}.encode_quantity_prompt",
                side_effect=fake_encode,
            ),
        ):
            install_quantity_prompt_unit(pipe, SimpleNamespace())
            unit = pipe.units[2]
            prompt = "motion at <extra_id_0>"
            inputs_shared = {"cfg_scale": 5.0}
            inputs_posi = {"prompt": prompt}
            inputs_nega = {"negative_prompt": prompt}
            with quantity_inference_conditioning(pipe, quantities):
                shared, positive, negative = unit.process(
                    pipe,
                    inputs_shared,
                    inputs_posi,
                    inputs_nega,
                )

        self.assertIs(shared, inputs_shared)
        self.assertIs(positive, inputs_posi)
        self.assertIs(negative, inputs_nega)
        self.assertTrue(unit.take_over)
        self.assertFalse(unit.seperate_cfg)
        self.assertEqual(
            [
                (prompt, quantities),
                (prompt, []),
            ],
            calls,
        )
        self.assertEqual(1.0, positive["context"].item())
        self.assertEqual(0.0, negative["context"].item())
        self.assertEqual(
            [{"name": "speed"}],
            pipe._last_quantity_token_audit,
        )
        self.assertEqual(
            [("text_encoder", "quantity_encoder")],
            loaded_models,
        )
        self.assertFalse(hasattr(pipe, "_active_quantity_conditioning"))

    def test_numeric_features_and_small_encoder_are_finite_and_trainable(
        self,
    ) -> None:
        self.assertEqual(8, len(NUMERIC_FEATURE_NAMES))
        self.assertEqual(
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            numeric_features(0.0),
        )
        positive = numeric_features(5.0)
        negative = numeric_features(-0.05)
        self.assertEqual(1.0, positive[1])
        self.assertAlmostEqual(math.log1p(5.0), positive[2])
        self.assertAlmostEqual(0.5, positive[5])
        self.assertEqual(0.0, positive[6])
        self.assertAlmostEqual(1.0 / 6.0, positive[7])
        self.assertEqual(-1.0, negative[1])
        self.assertAlmostEqual(-0.5, negative[5])
        self.assertAlmostEqual(-2.0 / 12.0, negative[6])
        self.assertTrue(all(math.isfinite(value) for value in positive))
        self.assertTrue(all(math.isfinite(value) for value in negative))
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    numeric_features(invalid)

        encoder = QuantityEncoder(_small_encoder_config())
        records = [
            {
                "si_value": 5.0,
                "dimension": [1, 0, -1, 0, 0, 0, 0],
                "quantity_type_id": 5,
            },
            {
                "si_value": math.pi / 6,
                "dimension": [0, 0, 0, 0, 0, 0, 0],
                "quantity_type_id": 3,
            },
        ]
        encoded = encoder.encode_records(records)
        self.assertEqual((2, 9), tuple(encoded.shape))
        self.assertEqual(torch.float32, encoded.dtype)
        self.assertTrue(torch.isfinite(encoded).all())
        weights = torch.arange(1, 10, dtype=encoded.dtype)
        (encoded * weights).sum().backward()
        self.assertTrue(any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and bool(torch.any(parameter.grad != 0))
            for parameter in encoder.parameters()
        ))

    def test_combined_checkpoint_splits_encoder_and_rejects_missing_weights(
        self,
    ) -> None:
        source = QuantityEncoder(_small_encoder_config())
        prefixed = {
            f"{QUANTITY_CHECKPOINT_PREFIX}{name}": tensor.detach().clone()
            for name, tensor in source.state_dict().items()
        }
        prefixed["pipe.dit.block.lora_A.weight"] = torch.ones(2, 2)
        modules = _fake_safetensors_modules(prefixed)
        with patch.dict(sys.modules, modules):
            extracted = quantity_state_dict("combined.safetensors")
            self.assertEqual(set(source.state_dict()), set(extracted))
            target = QuantityEncoder(_small_encoder_config())
            result = load_quantity_encoder_checkpoint(
                target,
                "combined.safetensors",
                required=True,
            )
        self.assertTrue(result["loaded"])
        self.assertEqual(len(source.state_dict()), result["tensor_count"])
        for name, expected in source.state_dict().items():
            self.assertTrue(torch.equal(expected, target.state_dict()[name]))

        lora_only = {
            "pipe.dit.block.lora_A.weight": torch.ones(2, 2),
            "pipe.dit.block.lora_B.weight": torch.ones(2, 2),
        }
        with patch.dict(
            sys.modules,
            _fake_safetensors_modules(lora_only),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "no quantity encoder tensors",
            ):
                load_quantity_encoder_checkpoint(
                    QuantityEncoder(_small_encoder_config()),
                    "lora_only.safetensors",
                    required=True,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            combined_path = root / "combined.safetensors"
            _write_safetensors_header(combined_path, {
                "pipe.dit.block.lora_A.weight": ("BF16", [2, 2]),
                "pipe.dit.block.lora_B.weight": ("BF16", [2, 2]),
                (
                    "pipe.quantity_encoder.numeric_mlp.0.weight"
                ): ("F32", [8, 8]),
            })
            inventory = Wan22QuantityLoraAdapter._checkpoint_inventory(
                combined_path
            )
            self.assertEqual(2, inventory["lora_tensor_count"])
            self.assertEqual(
                1,
                inventory["quantity_encoder_tensor_count"],
            )

            lora_path = root / "lora_only.safetensors"
            _write_safetensors_header(lora_path, {
                "pipe.dit.block.lora_A.weight": ("BF16", [2, 2]),
                "pipe.dit.block.lora_B.weight": ("BF16", [2, 2]),
            })
            with self.assertRaisesRegex(
                ValueError,
                "both LoRA and quantity encoder tensors",
            ):
                Wan22QuantityLoraAdapter._checkpoint_inventory(lora_path)

    def test_combined_checkpoint_strictly_validates_lora_before_fusion(
        self,
    ) -> None:
        source = QuantityEncoder(_small_encoder_config())
        quantity = {
            f"{QUANTITY_CHECKPOINT_PREFIX}{name}": tensor.detach().clone()
            for name, tensor in source.state_dict().items()
        }
        tensor_a = torch.tensor(
            [[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]],
        )
        tensor_b = torch.tensor(
            [
                [1.0, 2.0],
                [0.0, 1.0],
                [-1.0, 0.5],
                [2.0, -2.0],
            ],
        )
        lora = {
            "block.proj.lora_A.default.weight": tensor_a,
            "block.proj.lora_B.default.weight": tensor_b,
        }
        pipe = _FakeLoraPipe()
        weight_before = pipe.dit.block.proj.weight.detach().clone()
        target = QuantityEncoder(_small_encoder_config())
        with patch.dict(
            sys.modules,
            _fake_safetensors_modules({**quantity, **lora}),
        ):
            result = load_combined_quantity_checkpoint(
                pipe,
                target,
                "combined.safetensors",
                lora_alpha=0.5,
            )

        self.assertEqual(1, pipe.load_lora_calls)
        self.assertEqual(set(lora), pipe.loaded_lora_keys)
        self.assertEqual(2, result["dit_lora_tensor_count"])
        self.assertEqual(
            1,
            result["dit_lora_validated_layer_count"],
        )
        self.assertEqual(1, result["dit_lora_fused_layer_count"])
        self.assertTrue(torch.allclose(
            weight_before + 0.5 * torch.mm(tensor_b, tensor_a),
            pipe.dit.block.proj.weight,
        ))
        for name, expected in source.state_dict().items():
            self.assertTrue(
                torch.equal(expected, target.state_dict()[name])
            )

    def test_combined_checkpoint_rejects_incompatible_lora_without_fusion(
        self,
    ) -> None:
        source = QuantityEncoder(_small_encoder_config())
        quantity = {
            f"{QUANTITY_CHECKPOINT_PREFIX}{name}": tensor.detach().clone()
            for name, tensor in source.state_dict().items()
        }
        valid_a = torch.ones(2, 3)
        valid_b = torch.ones(4, 2)

        cases = {
            "missing_pair": (
                {"block.proj.lora_A.default.weight": valid_a},
                "not one-to-one",
            ),
            "extra_key": (
                {
                    "block.proj.lora_A.default.weight": valid_a,
                    "block.proj.lora_B.default.weight": valid_b,
                    "block.proj.alpha": torch.tensor(2.0),
                },
                "unsupported non-LoRA tensors",
            ),
            "unknown_target": (
                {
                    "block.absent.lora_A.default.weight": valid_a,
                    "block.absent.lora_B.default.weight": valid_b,
                },
                "target module does not exist",
            ),
            "rank_mismatch": (
                {
                    "block.proj.lora_A.default.weight": valid_a,
                    "block.proj.lora_B.default.weight": torch.ones(4, 3),
                },
                "rank mismatch",
            ),
            "target_shape_mismatch": (
                {
                    "block.proj.lora_A.default.weight": torch.ones(2, 5),
                    "block.proj.lora_B.default.weight": valid_b,
                },
                "shape does not match target weight",
            ),
        }
        for name, (lora, message) in cases.items():
            with self.subTest(name=name):
                pipe = _FakeLoraPipe()
                target = QuantityEncoder(_small_encoder_config())
                with patch.dict(
                    sys.modules,
                    _fake_safetensors_modules({**quantity, **lora}),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        load_combined_quantity_checkpoint(
                            pipe,
                            target,
                            "combined.safetensors",
                            lora_alpha=1.0,
                        )
                self.assertEqual(0, pipe.load_lora_calls)

    def test_finetune_dry_run_preserves_only_the_sealed_quantity_channel(
        self,
    ) -> None:
        value = copy.deepcopy(self.finetune_task.value)
        value["task_id"] = "free_fall_quantity_embedding_dry_run"
        value["selection"]["scene_ids"] = ["free_fall"]
        value["selection"]["eval_partitions"] = ["test_id"]
        task = TaskSpec(
            self.finetune_task.path,
            value,
            canonical_sha256(value),
        )
        instance = self.plugin.task_builder.build(self.dataset, task)
        self.assertEqual(7, len(instance.canonical_plan.train_case_ids))
        self.assertEqual(4, len(instance.canonical_plan.jobs))

        def fake_video(
            _media,
            source,
            output,
            *,
            materialize,
            speed_factor=1.0,
            scene_id=None,
        ):
            del materialize, speed_factor, scene_id
            return {
                "source": str(source),
                "output": str(output),
                "target_frames": 5,
                "generation_target_frames": 5,
            }

        def fake_first_frame(
            _media,
            source,
            output,
            *,
            source_is_video,
            materialize,
            scene_id=None,
        ):
            del source_is_video, materialize, scene_id
            return {"source": str(source), "output": str(output)}

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _create_run_directories(run_dir)
            with (
                patch.object(
                    Wan22MediaAdapter,
                    "normalize_video",
                    autospec=True,
                    side_effect=fake_video,
                ),
                patch.object(
                    Wan22MediaAdapter,
                    "normalize_first_frame",
                    autospec=True,
                    side_effect=fake_first_frame,
                ),
            ):
                training, predictions = self.plugin.run_task(
                    instance=instance,
                    run_dir=run_dir,
                    execute=False,
                    stop_after_training=False,
                )

            self.assertEqual("planned", training["status"])
            metadata = load_jsonl(training["metadata"])
            self.assertEqual(7, len(metadata))
            metadata_fields = {
                "video",
                "prompt",
                "audited_prompt",
                "quantities",
                "quantity_registry_id",
                "quantity_registry_fingerprint",
                "case_id",
                "scene_id",
                "text_transform_id",
            }
            train_adaptations = {
                adaptation["case_id"]: adaptation
                for adaptation in instance.value["adaptations"]
                if adaptation["role"] == "train"
            }
            for row in metadata:
                self.assertEqual(metadata_fields, set(row))
                native = train_adaptations[row["case_id"]][
                    "native_inputs"
                ]
                self.assertEqual(native["text"]["prompt"], row["prompt"])
                self.assertEqual(
                    native["text"]["audited_prompt"],
                    row["audited_prompt"],
                )
                self.assertEqual(
                    native["physics"]["quantities"],
                    row["quantities"],
                )
                self.assertNotIn("physical_parameters", row)

            frozen_jobs = {
                job["job_id"]: job
                for job in instance.value["inference"]["jobs"]
            }
            self.assertEqual(set(frozen_jobs), {
                prediction["job_id"] for prediction in predictions
            })
            self.assertTrue(all(
                prediction["status"] == "planned"
                and prediction["video_path"] is None
                for prediction in predictions
            ))
            for job_id, frozen in frozen_jobs.items():
                prepared = load_json(run_dir / "jobs" / f"{job_id}.json")
                model_input = prepared["model_input"]
                self.assertEqual(
                    {
                        "prompt",
                        "first_frame",
                        "audited_prompt",
                        "quantity_payload",
                    },
                    set(model_input),
                )
                self.assertEqual(
                    frozen["native_inputs"]["physics"],
                    model_input["quantity_payload"],
                )
                self.assertEqual(
                    frozen["native_inputs"]["text"]["prompt"],
                    model_input["prompt"],
                )
                self.assertNotIn("physical_parameters", model_input)
                self.assertNotIn(
                    "physical_parameters",
                    prepared["adaptation"]["native_inputs"],
                )


if __name__ == "__main__":
    unittest.main()
