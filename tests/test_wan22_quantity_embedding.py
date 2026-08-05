from __future__ import annotations

import copy
import json
import math
import os
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
    QUANTITY_ENCODER_TENSOR_COUNT,
    QUANTITY_CHECKPOINT_PREFIX,
    QuantityEncoder,
    WAN22_TI2V_5B_LORA_PAIR_COUNT,
    WAN22_TI2V_5B_LORA_RANK,
    WAN22_TI2V_5B_LORA_TENSOR_COUNT,
    distributed_sampling_contract,
    expected_wan22_ti2v_5b_lora_targets,
    install_quantity_prompt_unit,
    load_combined_quantity_checkpoint,
    load_quantity_encoder_checkpoint,
    load_verified_combined_quantity_checkpoint,
    locate_sentinel_tokens,
    numeric_features,
    quantity_inference_conditioning,
    quantity_pipeline_shared_config,
    quantity_pipeline_shared_fingerprint,
    quantity_state_dict,
    verify_quantity_checkpoint_manifest,
)
from physbench.data_layout import V4_DATASET
from physbench.datasets import load_dataset
from physbench.domain import TaskSpec
from physbench.io import (
    canonical_sha256,
    load_json,
    load_jsonl,
    sha256_file,
    write_json,
)


BASELINE = (
    ROOT / "baselines" / "wan22_quantity_embedding" / "baseline.json"
)
QUANTITY_FIELDS = {
    "name",
    "symbol",
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
    *,
    load_bytes=None,
) -> dict[str, ModuleType]:
    package = ModuleType("safetensors")
    torch_module = ModuleType("safetensors.torch")
    torch_module.load_file = lambda _path, device="cpu": {  # type: ignore[attr-defined]
        key: value.to(device)
        for key, value in state.items()
    }
    torch_module.load = (  # type: ignore[attr-defined]
        load_bytes
        if load_bytes is not None
        else lambda _data: {
            key: value.detach().clone()
            for key, value in state.items()
        }
    )
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
    path.write_bytes(
        struct.pack("<Q", len(payload))
        + payload
        + (b"\0" * offset)
    )


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


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in ("q", "k", "v", "o"):
            setattr(self, name, nn.Linear(3, 4, bias=False))


class _FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()
        self.cross_attn = _FakeAttention()
        self.ffn = nn.Sequential(
            nn.Linear(3, 4, bias=False),
            nn.Identity(),
            nn.Linear(3, 4, bias=False),
        )


class _FakeDit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_FakeBlock() for _ in range(30)]
        )


def _full_lora_state(
    pipe: "_FakeLoraPipe",
    *,
    rank: int = WAN22_TI2V_5B_LORA_RANK,
) -> dict[str, torch.Tensor]:
    modules = dict(pipe.dit.named_modules())
    state = {}
    for target in sorted(expected_wan22_ti2v_5b_lora_targets()):
        weight = modules[target].weight
        state[f"{target}.lora_A.default.weight"] = torch.full(
            (rank, int(weight.shape[1])),
            0.01,
        )
        state[f"{target}.lora_B.default.weight"] = torch.full(
            (int(weight.shape[0]), rank),
            0.02,
        )
    return state


def _full_checkpoint_header(
    pipe: "_FakeLoraPipe",
    encoder: QuantityEncoder,
) -> dict[str, tuple[str, list[int]]]:
    header = {
        f"{QUANTITY_CHECKPOINT_PREFIX}{name}": (
            "F32",
            list(tensor.shape),
        )
        for name, tensor in encoder.state_dict().items()
    }
    header.update({
        name: ("BF16", list(tensor.shape))
        for name, tensor in _full_lora_state(pipe).items()
    })
    return header


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

    def test_all_214_cases_adapt_without_task_compilation(self) -> None:
        self.assertEqual("1.0.1", self.bundle.value["baseline_version"])
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
                    {
                        "vision",
                        "text",
                        "generation_shape",
                        "physics",
                        "media_contract",
                    },
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
                    self.assertEqual(source.get("symbol"), quantity["symbol"])
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

    def test_adapter_rejects_non_finite_annotated_values(self) -> None:
        case = copy.deepcopy(self.dataset.cases[0])
        parameter = next(
            item
            for item in self.adapter.registry.value["scenes"][
                case["scene_id"]
            ]["parameters"]
            if (
                isinstance(case["physics"].get(item["name"]), dict)
                and case["physics"][item["name"]].get("annotated") is True
            )
        )
        for value in (
            float("nan"),
            float("inf"),
            float("-inf"),
            10**1000,
        ):
            with self.subTest(value=value):
                invalid = copy.deepcopy(case)
                invalid["physics"][parameter["name"]]["value"] = value
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    self.adapter.adapt_case(invalid, role="eval")

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
        self.assertEqual(
            len(native["physics"]["quantities"]), len(audits)
        )
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
        invalid_dimension = copy.deepcopy(records[:1])
        invalid_dimension[0]["dimension"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "integer exponents"):
            encoder.encode_records(invalid_dimension)
        invalid_type = copy.deepcopy(records[:1])
        invalid_type[0]["quantity_type_id"] = True
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            encoder.encode_records(invalid_type)
        overflowing = QuantityEncoder(_small_encoder_config()).eval()
        with torch.no_grad():
            for parameter in overflowing.parameters():
                parameter.fill_(torch.finfo(parameter.dtype).max)
        with self.assertRaisesRegex(
            FloatingPointError,
            "non-finite physical embeddings",
        ):
            overflowing.encode_records(records[:1])

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
            source = QuantityEncoder(_small_encoder_config())
            pipe = _FakeLoraPipe()
            header = _full_checkpoint_header(pipe, source)
            _write_safetensors_header(combined_path, header)
            inventory = Wan22QuantityLoraAdapter._checkpoint_inventory(
                combined_path
            )
            self.assertTrue(inventory["safetensors_layout_verified"])
            self.assertTrue(inventory["finite_payload_verified"])
            self.assertEqual(
                WAN22_TI2V_5B_LORA_TENSOR_COUNT,
                inventory["lora_tensor_count"],
            )
            self.assertEqual(
                WAN22_TI2V_5B_LORA_PAIR_COUNT,
                inventory["lora_pair_count"],
            )
            self.assertEqual(
                WAN22_TI2V_5B_LORA_RANK,
                inventory["lora_rank"],
            )
            self.assertEqual(
                QUANTITY_ENCODER_TENSOR_COUNT,
                inventory["quantity_encoder_tensor_count"],
            )

            non_finite_path = root / "non_finite.safetensors"
            _write_safetensors_header(non_finite_path, header)
            with non_finite_path.open("r+b") as handle:
                header_size = struct.unpack("<Q", handle.read(8))[0]
                header_value = json.loads(handle.read(header_size))
                quantity_key = next(
                    key
                    for key, value in header_value.items()
                    if key.startswith(QUANTITY_CHECKPOINT_PREFIX)
                    and value["dtype"] == "F32"
                    and math.prod(value["shape"]) > 0
                )
                handle.seek(
                    8
                    + header_size
                    + header_value[quantity_key]["data_offsets"][0]
                )
                handle.write(struct.pack("<f", float("inf")))
            with self.assertRaisesRegex(
                ValueError,
                "contains non-finite values",
            ):
                Wan22QuantityLoraAdapter._checkpoint_inventory(
                    non_finite_path
                )

            lora_path = root / "lora_only.safetensors"
            _write_safetensors_header(lora_path, {
                name: ("BF16", list(tensor.shape))
                for name, tensor in _full_lora_state(pipe).items()
            })
            with self.assertRaisesRegex(
                ValueError,
                "exactly 619 tensors",
            ):
                Wan22QuantityLoraAdapter._checkpoint_inventory(lora_path)

            corruptions = {}
            missing_quantity = dict(header)
            missing_quantity.pop(next(
                key
                for key in missing_quantity
                if key.startswith(QUANTITY_CHECKPOINT_PREFIX)
            ))
            corruptions["quantity_count"] = (
                missing_quantity,
                "exactly 619 tensors",
            )
            bad_rank = dict(header)
            rank_key = next(
                key for key in bad_rank if ".lora_A." in key
            )
            dtype, shape = bad_rank[rank_key]
            bad_rank[rank_key] = (dtype, [31, shape[1]])
            corruptions["rank"] = (bad_rank, "rank must be exactly 32")
            bad_target = dict(header)
            target_key_a = next(
                key for key in bad_target if ".lora_A." in key
            )
            target_key_b = target_key_a.replace(".lora_A.", ".lora_B.")
            replacement_a = target_key_a.replace(
                "blocks.0.",
                "blocks.30.",
                1,
            )
            replacement_b = target_key_b.replace(
                "blocks.0.",
                "blocks.30.",
                1,
            )
            bad_target[replacement_a] = bad_target.pop(target_key_a)
            bad_target[replacement_b] = bad_target.pop(target_key_b)
            corruptions["target"] = (
                bad_target,
                "target topology mismatch",
            )
            for name, (corrupt, message) in corruptions.items():
                with self.subTest(inventory=name):
                    candidate = root / f"{name}.safetensors"
                    _write_safetensors_header(candidate, corrupt)
                    with self.assertRaisesRegex(ValueError, message):
                        Wan22QuantityLoraAdapter._checkpoint_inventory(
                            candidate
                        )

    def test_combined_checkpoint_strictly_validates_lora_before_fusion(
        self,
    ) -> None:
        source = QuantityEncoder(_small_encoder_config())
        quantity = {
            f"{QUANTITY_CHECKPOINT_PREFIX}{name}": tensor.detach().clone()
            for name, tensor in source.state_dict().items()
        }
        pipe = _FakeLoraPipe()
        lora = _full_lora_state(pipe)
        target_name = "blocks.0.self_attn.q"
        tensor_a = lora[
            f"{target_name}.lora_A.default.weight"
        ]
        tensor_b = lora[
            f"{target_name}.lora_B.default.weight"
        ]
        weight_before = (
            dict(pipe.dit.named_modules())[target_name]
            .weight.detach()
            .clone()
        )
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
        self.assertEqual(
            WAN22_TI2V_5B_LORA_TENSOR_COUNT,
            result["dit_lora_tensor_count"],
        )
        self.assertEqual(
            WAN22_TI2V_5B_LORA_PAIR_COUNT,
            result["dit_lora_validated_layer_count"],
        )
        self.assertEqual(
            WAN22_TI2V_5B_LORA_PAIR_COUNT,
            result["dit_lora_fused_layer_count"],
        )
        self.assertTrue(torch.allclose(
            weight_before + 0.5 * torch.mm(tensor_b, tensor_a),
            dict(pipe.dit.named_modules())[target_name].weight,
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
        target_name = "blocks.0.self_attn.q"
        key_a = f"{target_name}.lora_A.default.weight"
        key_b = f"{target_name}.lora_B.default.weight"
        cases = {}
        pipe = _FakeLoraPipe()
        missing_pair = _full_lora_state(pipe)
        del missing_pair[key_b]
        cases["missing_pair"] = (missing_pair, "exactly 600")

        extra_key = _full_lora_state(pipe)
        extra_key["blocks.0.self_attn.q.alpha"] = torch.tensor(2.0)
        cases["extra_key"] = (
            extra_key,
            "unsupported non-LoRA tensors",
        )

        unknown_target = _full_lora_state(pipe)
        unknown_target[
            "blocks.30.self_attn.q.lora_A.default.weight"
        ] = unknown_target.pop(key_a)
        unknown_target[
            "blocks.30.self_attn.q.lora_B.default.weight"
        ] = unknown_target.pop(key_b)
        cases["unknown_target"] = (
            unknown_target,
            "target topology",
        )

        rank_mismatch = _full_lora_state(pipe)
        rank_mismatch[key_b] = torch.ones(4, 31)
        cases["rank_mismatch"] = (
            rank_mismatch,
            "rank must be exactly 32",
        )

        target_shape_mismatch = _full_lora_state(pipe)
        target_shape_mismatch[key_a] = torch.ones(32, 5)
        cases["target_shape_mismatch"] = (
            target_shape_mismatch,
            "shape does not match target weight",
        )

        non_finite = _full_lora_state(pipe)
        non_finite[key_a][0, 0] = float("nan")
        cases["non_finite"] = (
            non_finite,
            "contains non-finite values",
        )

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

        pipe = _FakeLoraPipe()
        with patch.dict(
            sys.modules,
            _fake_safetensors_modules({
                **quantity,
                **_full_lora_state(pipe),
            }),
        ):
            with self.assertRaisesRegex(ValueError, "alpha must be finite"):
                load_combined_quantity_checkpoint(
                    pipe,
                    QuantityEncoder(_small_encoder_config()),
                    "combined.safetensors",
                    lora_alpha=float("nan"),
                )
        self.assertEqual(0, pipe.load_lora_calls)

    def test_combined_checkpoint_rejects_quantity_topology_and_nan_before_fusion(
        self,
    ) -> None:
        source = QuantityEncoder(_small_encoder_config())
        quantity = {
            f"{QUANTITY_CHECKPOINT_PREFIX}{name}": tensor.detach().clone()
            for name, tensor in source.state_dict().items()
        }
        pipe = _FakeLoraPipe()
        lora = _full_lora_state(pipe)

        missing = dict(quantity)
        missing.pop(next(iter(missing)))
        non_finite = {
            key: tensor.detach().clone()
            for key, tensor in quantity.items()
        }
        first_key = next(iter(non_finite))
        non_finite[first_key].flatten()[0] = float("inf")
        mixed_prefix = dict(quantity)
        mixed_key = next(iter(mixed_prefix))
        mixed_suffix = mixed_key.removeprefix(
            QUANTITY_CHECKPOINT_PREFIX
        )
        mixed_prefix[
            f"quantity_encoder.{mixed_suffix}"
        ] = mixed_prefix.pop(mixed_key)
        for name, (candidate, message) in {
            "missing": (missing, "exactly 19"),
            "non_finite": (non_finite, "contains non-finite values"),
            "mixed_prefix": (mixed_prefix, "mixes quantity encoder"),
        }.items():
            with self.subTest(name=name):
                candidate_pipe = _FakeLoraPipe()
                target = QuantityEncoder(_small_encoder_config())
                before = {
                    key: tensor.detach().clone()
                    for key, tensor in target.state_dict().items()
                }
                with patch.dict(
                    sys.modules,
                    _fake_safetensors_modules({**candidate, **lora}),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        load_combined_quantity_checkpoint(
                            candidate_pipe,
                            target,
                            "combined.safetensors",
                            lora_alpha=1.0,
                        )
                self.assertEqual(0, candidate_pipe.load_lora_calls)
                for key, expected in before.items():
                    self.assertTrue(
                        torch.equal(expected, target.state_dict()[key])
                    )

    def test_checkpoint_manifest_verification_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "step-1.safetensors"
            checkpoint.write_bytes(b"sealed checkpoint bytes")
            manifest = root / "checkpoint.json"
            value = {
                "schema_version": "2.0",
                "status": "complete",
                "checkpoint": str(checkpoint),
                "checkpoint_size": checkpoint.stat().st_size,
                "checkpoint_sha256": sha256_file(checkpoint),
            }
            write_json(manifest, value)
            audit = verify_quantity_checkpoint_manifest(
                checkpoint,
                manifest,
            )
            self.assertTrue(audit["verified"])
            self.assertEqual(
                "single_fd_single_bytes",
                audit["checkpoint_read_mode"],
            )
            self.assertEqual(
                value["checkpoint_sha256"],
                audit["checkpoint_sha256"],
            )

            invalid_values = {
                "size": {
                    **value,
                    "checkpoint_size": value["checkpoint_size"] + 1,
                },
                "digest": {
                    **value,
                    "checkpoint_sha256": "0" * 64,
                },
                "path": {
                    **value,
                    "checkpoint": str(root / "different.safetensors"),
                },
                "status": {
                    **value,
                    "status": "planned",
                },
                "schema": {
                    **value,
                    "schema_version": "1.0",
                },
                "missing_size": {
                    key: item
                    for key, item in value.items()
                    if key != "checkpoint_size"
                },
                "missing_digest": {
                    key: item
                    for key, item in value.items()
                    if key != "checkpoint_sha256"
                },
            }
            for name, invalid in invalid_values.items():
                with self.subTest(name=name):
                    write_json(manifest, invalid)
                    with self.assertRaises(
                        (ValueError, FileNotFoundError)
                    ):
                        verify_quantity_checkpoint_manifest(
                            checkpoint,
                            manifest,
                        )
            with self.assertRaises(FileNotFoundError):
                verify_quantity_checkpoint_manifest(
                    checkpoint,
                    root / "missing.json",
                )
            write_json(manifest, value)
            symlink = root / "checkpoint-link.safetensors"
            symlink.symlink_to(checkpoint)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                verify_quantity_checkpoint_manifest(
                    symlink,
                    manifest,
                )

    def test_verified_checkpoint_loader_binds_hash_and_parser_to_same_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.safetensors"
            replacement = root / "replacement.safetensors"
            backup = root / "sealed.backup"
            manifest = root / "checkpoint.json"
            source = QuantityEncoder(_small_encoder_config())
            pipe = _FakeLoraPipe()
            _write_safetensors_header(
                checkpoint,
                _full_checkpoint_header(pipe, source),
            )
            original_bytes = checkpoint.read_bytes()
            replacement.write_bytes(
                original_bytes[:-1]
                + bytes([original_bytes[-1] ^ 1])
            )
            write_json(manifest, {
                "schema_version": "2.0",
                "status": "complete",
                "checkpoint": str(checkpoint),
                "checkpoint_size": checkpoint.stat().st_size,
                "checkpoint_sha256": sha256_file(checkpoint),
            })
            original_state = {
                **{
                    f"{QUANTITY_CHECKPOINT_PREFIX}{name}": (
                        tensor.detach().clone()
                    )
                    for name, tensor in source.state_dict().items()
                },
                **_full_lora_state(pipe),
            }
            malicious_state = {
                key: tensor.detach().clone()
                for key, tensor in original_state.items()
            }
            quantity_key = next(
                key
                for key in malicious_state
                if key.startswith(QUANTITY_CHECKPOINT_PREFIX)
            )
            malicious_state[quantity_key].flatten()[0].add_(123.0)
            observed = {}

            def load_same_bytes(data):
                os.rename(checkpoint, backup)
                os.rename(replacement, checkpoint)
                try:
                    observed["path_bytes_during_parse"] = (
                        checkpoint.read_bytes()
                    )
                    observed["parser_bytes"] = data
                    selected = (
                        original_state
                        if data == original_bytes
                        else malicious_state
                    )
                    return {
                        key: tensor.detach().clone()
                        for key, tensor in selected.items()
                    }
                finally:
                    os.rename(checkpoint, replacement)
                    os.rename(backup, checkpoint)

            target = QuantityEncoder(_small_encoder_config())
            modules = _fake_safetensors_modules(
                original_state,
                load_bytes=load_same_bytes,
            )
            with patch.dict(sys.modules, modules):
                loaded = load_verified_combined_quantity_checkpoint(
                    pipe,
                    target,
                    checkpoint,
                    manifest,
                    lora_alpha=1.0,
                )
            self.assertTrue(loaded["load_boundary_verified"])
            self.assertEqual(
                "manifest_hash_and_safetensors_same_bytes",
                loaded["checkpoint_load_mode"],
            )
            self.assertTrue(
                loaded["checkpoint_verification"]["verified"]
            )
            self.assertEqual(
                original_bytes,
                observed["parser_bytes"],
            )
            self.assertNotEqual(
                original_bytes,
                observed["path_bytes_during_parse"],
            )
            self.assertEqual(original_bytes, checkpoint.read_bytes())
            loaded_key = quantity_key.removeprefix(
                QUANTITY_CHECKPOINT_PREFIX
            )
            self.assertTrue(torch.equal(
                target.state_dict()[loaded_key],
                original_state[quantity_key],
            ))
            self.assertFalse(torch.equal(
                target.state_dict()[loaded_key],
                malicious_state[quantity_key],
            ))

    def test_pipeline_shared_config_catches_load_time_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.safetensors"
            manifest = root / "checkpoint.json"
            job = {
                "checkpoint": str(checkpoint),
                "checkpoint_manifest": str(manifest),
                "wan22": {
                    "runtime": {
                        "project_root": "/runtime",
                        "model_base": "/models",
                    },
                    "quantity_encoder": _small_encoder_config(),
                    "generation": {
                        "lora_alpha": 1.0,
                        "height": 480,
                        "width": 832,
                        "num_frames": 121,
                    },
                },
            }
            original = quantity_pipeline_shared_config(job)
            per_call_change = copy.deepcopy(job)
            per_call_change["wan22"]["generation"]["height"] = 832
            self.assertEqual(
                original,
                quantity_pipeline_shared_config(per_call_change),
            )
            alpha_change = copy.deepcopy(job)
            alpha_change["wan22"]["generation"]["lora_alpha"] = 0.5
            self.assertNotEqual(
                original,
                quantity_pipeline_shared_config(alpha_change),
            )
            runtime_change = copy.deepcopy(job)
            runtime_change["wan22"]["runtime"]["model_base"] = "/other"
            self.assertNotEqual(
                original,
                quantity_pipeline_shared_config(runtime_change),
            )
            self.assertEqual(
                canonical_sha256(original),
                quantity_pipeline_shared_fingerprint(job),
            )
            bad_alpha = copy.deepcopy(job)
            bad_alpha["wan22"]["generation"]["lora_alpha"] = float("nan")
            with self.assertRaisesRegex(ValueError, "alpha must be finite"):
                quantity_pipeline_shared_config(bad_alpha)

    def test_distributed_sampling_contract_forbids_even_batch_padding(
        self,
    ) -> None:
        contract = distributed_sampling_contract(1160, 8)
        self.assertEqual(145, contract["samples_per_rank_per_epoch"])
        self.assertFalse(contract["even_batch_padding_required"])
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            distributed_sampling_contract(28, 8)

    def test_finetune_dry_run_preserves_only_the_sealed_quantity_channel(
        self,
    ) -> None:
        value = {
            "schema_version": "3.0",
            "task_id": "free_fall_quantity_embedding_dry_run",
            "family": "finetune_eval",
            "dataset_id": self.dataset.dataset_id,
            "dataset_view": "view_a",
            "selection": {
                "scene_ids": ["free_fall"],
                "eval_partitions": ["test_id"],
            },
            "ood2": {"enabled": False},
            "seeds": {"training": [42], "inference": [42]},
            "evaluation": {"protocol": "scene_default_v3"},
        }
        task = TaskSpec(
            ROOT / "tests" / "fixtures" / "legacy_v4_finetune_task.json",
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
            sampling_plan = load_json(
                run_dir
                / "artifacts"
                / "wan22"
                / "training_sampling_plan.json"
            )
            self.assertTrue(sampling_plan["shuffle"])
            self.assertEqual(42, sampling_plan["sampler_seed"])
            self.assertEqual(
                "torch.Generator",
                sampling_plan["sampler_generator"],
            )
            self.assertEqual(
                "DataLoader(generator=...)",
                sampling_plan["sampler_binding"],
            )
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
            predictions_by_id = {
                prediction["job_id"]: prediction
                for prediction in predictions
            }
            for job_id, frozen in frozen_jobs.items():
                self.assertEqual(
                    frozen["scene_id"],
                    predictions_by_id[job_id]["scene_id"],
                )
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

    def test_training_entrypoint_seals_sampler_seed_in_all_audits(
        self,
    ) -> None:
        training_source = (
            ROOT / "scripts" / "wan22_quantity_train.py"
        ).read_text(encoding="utf-8")
        launcher_source = (
            ROOT / "scripts" / "train_wan22_quantity.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("generator=sampler_generator", training_source)
        self.assertIn('"--sampler_seed"', training_source)
        self.assertIn(
            '"training_sampling_runtime.json"',
            training_source,
        )
        self.assertIn(
            "distributed_sampling_contract(",
            training_source,
        )
        self.assertIn(
            '"sampler_generator": self.sampler_generator.get_state()',
            training_source,
        )
        self.assertIn(
            '"sampler_generator_state_checkpointed": bool(',
            training_source,
        )
        self.assertIn(
            "assert_loss_finite(accelerator, loss)",
            training_source,
        )
        self.assertIn(
            "assert_quantity_gradients_finite(",
            training_source,
        )
        self.assertIn(
            'echo "sampler_seed=$TRAIN_SEED"',
            launcher_source,
        )
        self.assertIn(
            '--sampler_seed "$TRAIN_SEED"',
            launcher_source,
        )

    def test_single_and_batch_generation_share_checkpoint_audits(
        self,
    ) -> None:
        single_source = (
            ROOT / "scripts" / "wan22_quantity_generate.py"
        ).read_text(encoding="utf-8")
        batch_source = (
            ROOT / "scripts" / "wan22_quantity_generate_batch.py"
        ).read_text(encoding="utf-8")
        adapter_source = (
            ROOT
            / "src"
            / "physbench"
            / "baselines"
            / "wan22_quantity.py"
        ).read_text(encoding="utf-8")
        for source in (single_source, batch_source):
            self.assertIn(
                "load_verified_combined_quantity_checkpoint(",
                source,
            )
            self.assertIn('"scene_id": job["scene_id"]', source)
            self.assertIn(
                '"pipeline_shared_config_fingerprint"',
                source,
            )
            self.assertIn('"load_boundary_verified"', source)
        self.assertIn(
            "quantity_pipeline_shared_config(job)",
            batch_source,
        )
        for field in (
            '"scene_id"',
            '"quantity_token_audit"',
            '"quantity_count"',
            '"checkpoint_sha256"',
            '"checkpoint_size"',
            '"checkpoint_manifest"',
            '"load_boundary_verified"',
            '"checkpoint_load_mode"',
            '"pipeline_shared_config_fingerprint"',
        ):
            self.assertIn(field, batch_source)
            self.assertIn(field, adapter_source)


if __name__ == "__main__":
    unittest.main()
